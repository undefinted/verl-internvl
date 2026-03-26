# Copyright 2025 HuggingFace Inc. team & Bytedance Ltd. and/or its affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.internvl.modeling_internvl import (
    InternVLForConditionalGeneration,
    InternVLModel,
    InternVLCausalLMOutputWithPast,
    InternVLPreTrainedModel,
    InternVLVisionAttention,
    InternVLVisionLayer,
    InternVLModelOutputWithPast,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)


def add_diffusion_noise(image_tensor, noise_step, gamma=0.005):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device, dtype=image_tensor.dtype)
    betas = torch.sigmoid(betas) * (gamma - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    return image_tensor_cd


def GMM_mask(
    sig,
    thres_mode,
    valid_mask=None,
    head_wise=False,
):
    data = sig

    if valid_mask is None:
        valid_mask = data > 1e-6

    valid_data = data[valid_mask]

    if valid_data.numel() == 0:
        return torch.zeros_like(data), torch.tensor(0.0, device=data.device), torch.tensor(0.0, device=data.device)

    if head_wise:
        if valid_mask.ndim < data.ndim:
            valid_mask = valid_mask.expand_as(data)
        valid_count = valid_mask.sum(dim=(0, 2, 3), keepdim=True).clamp(min=1.0)
        valid_sum = (data * valid_mask.float()).sum(dim=(0, 2, 3), keepdim=True)
        mean = valid_sum / valid_count
        valid_sum_sq = ((data**2) * valid_mask.float()).sum(dim=(0, 2, 3), keepdim=True)
        mean_sq = valid_sum_sq / valid_count
        std = torch.sqrt((mean_sq - mean**2).clamp(min=1e-10))
    else:
        mean = torch.mean(valid_data)
        std = torch.std(valid_data)

    if thres_mode == "high":
        thres = mean + 2 * std
    elif thres_mode == "medium":
        thres = mean + std
    elif thres_mode == "low":
        thres = mean
    elif thres_mode == "extra":
        thres = mean + 3 * std
    else:
        raise ValueError(f"Unknown thres_mode: {thres_mode}")

    mask = (sig > thres).float()

    if valid_mask is not None:
        mask = mask * valid_mask.float()

    return mask, mean, std


def values_noise_multiply(attn_weights, image_pos, value_states, thres_mode, noise, mean_mode):
    bsz, num_heads, q_len, k_len = attn_weights.shape
    head_dim = value_states.shape[-1]

    if noise:
        noise_value_states = add_diffusion_noise(value_states.clone(), 999, 0.01)
    else:
        if mean_mode == "text":
            value_mask = torch.ones((bsz, k_len), dtype=torch.bool, device=value_states.device)
        elif mean_mode == "image":
            value_mask = torch.zeros((bsz, k_len), dtype=torch.bool, device=value_states.device)
        else:
            raise ValueError(f"Unknown mean_mode: {mean_mode}")

        num_tokens_per_item = torch.zeros((bsz, 1), dtype=value_states.dtype, device=value_states.device)
        for i, intervals in enumerate(image_pos):
            for interval in intervals:
                start, end = interval["image_token_start"], interval["image_token_end"]
                if start < k_len and end <= k_len:
                    if mean_mode == "text":
                        value_mask[i, start:end] = False
                    else:
                        value_mask[i, start:end] = True
            count = value_mask[i].sum()
            num_tokens_per_item[i] = max(count, torch.tensor(1.0, dtype=value_states.dtype, device=value_states.device))

        value_mask_expanded = value_mask.view(bsz, 1, k_len, 1).expand_as(value_states)
        masked_values = value_states.where(
            value_mask_expanded, torch.tensor(0.0, dtype=value_states.dtype, device=value_states.device)
        )
        mean = torch.sum(masked_values, dim=(2, 3), keepdim=True) / (
            num_tokens_per_item.view(bsz, 1, 1, 1) * head_dim
        )
        noise_value_states = mean.expand_as(value_states)

    final_mask = torch.zeros_like(attn_weights)
    for i, intervals in enumerate(image_pos):
        for interval in intervals:
            start, end = interval["image_token_start"], interval["image_token_end"]
            if q_len > end and start < end:
                img_attn_slice = attn_weights[i : i + 1, :, end:, start:end]
                if img_attn_slice.numel() > 0:
                    valid_slice_mask = img_attn_slice > 1e-8
                    if valid_slice_mask.any():
                        mask, _, _ = GMM_mask(img_attn_slice, valid_mask=valid_slice_mask, thres_mode=thres_mode)
                        final_mask[i : i + 1, :, end:, start:end] = mask

    final_mask = final_mask.to(value_states.dtype)
    return final_mask, noise_value_states


def internvl_attn_forward(
    self: InternVLVisionAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    apply_cmve: bool = False,
    image_pos: Optional[List[dict]] = None,
    thres_mode: str = None,
    noise: bool = False,
    mean_mode: str = None,
    **kwargs,
):
    """
    替换语言模型中 attention 的 forward：
    - apply_cmve = False: 复用官方 ALL_ATTENTION_FUNCTIONS（与 modeling_internvl 一致）
    - apply_cmve = True: 在标准注意力上叠加 CMVE 的增量项
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if use_cache:
        present_key_value = (key_states, value_states)
    else:
        present_key_value = None

    if apply_cmve:
        # 手写 full-attention，用于构造 CMVE
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        final_mask, ave_value_states = values_noise_multiply(
            attn_weights, image_pos, value_states, thres_mode, noise, mean_mode
        )

        delta_v = ave_value_states - value_states
        attn_output_delta = torch.matmul(attn_weights * final_mask, delta_v)
        attn_output_delta = attn_output_delta.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        attn_output_ori = torch.matmul(attn_weights, value_states)
        attn_output_ori = attn_output_ori.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        attn_output = attn_output_ori + attn_output_delta

    else:
        # 非 CMVE 时，复用官方 attention backend，与 modeling_internvl.InternVLVisionAttention.forward 对齐
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

    attn_output = self.projection_layer(attn_output)
    attn_output = self.projection_dropout(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, present_key_value


def internvl_decoder_forward_new(
    self: InternVLVisionLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    apply_cmve: bool = False,
    image_pos: Optional[List[dict]] = None,
    thres_mode: str = None,
    noise: bool = False,
    mean_mode: str = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    与官方 InternVLVisionLayer.forward 拓扑一致，只是把 CMVE 相关参数往下传给 attention。
    """
    residual = hidden_states
    hidden_states = self.layernorm_before(hidden_states)

    attention_output, attn_weights, present_key_value = self.attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        apply_cmve=apply_cmve,
        image_pos=image_pos,
        thres_mode=thres_mode,
        noise=noise,
        mean_mode=mean_mode,
        **kwargs,
    )

    attention_output = self.lambda_1 * attention_output
    hidden_states = attention_output + residual

    residual = hidden_states
    hidden_states = self.layernorm_after(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.dropout(hidden_states)

    if self.lambda_2 is not None:
        hidden_states = self.lambda_2 * hidden_states

    hidden_states = hidden_states + residual

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    if use_cache:
        outputs += (present_key_value,)

    return outputs


def internvl_language_forward_new(
    self: InternVLPreTrainedModel,
    inputs_embeds: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    apply_cmve: bool = False,
    image_pos: Optional[List[dict]] = None,
    thres_mode: str = None,
    noise: bool = False,
    mean_mode: str = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """
    语言模型的 CMVE 版 forward，基本仿照 transformers 的 Llama/Qwen decoder 写法。
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            apply_cmve=apply_cmve,
            image_pos=image_pos,
            thres_mode=thres_mode,
            noise=noise,
            mean_mode=mean_mode,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def internvl_language_forward_original(
    self: InternVLModel,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    vision_feature_layer: int | List[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    **kwargs,
) -> InternVLModelOutputWithPast:
    """
    封装官方 InternVLModel.forward 的逻辑，用于 cmve=0 时保持完全一致的行为。
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_features = None
    if pixel_values is not None:
        vision_outputs = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        )
        image_features = vision_outputs.pooler_output
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return InternVLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def internvl_base_forward_new(
    self: InternVLModel,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    apply_cmve: bool = False,
    image_pos: Optional[List[dict]] = None,
    thres_mode: str = None,
    noise: bool = False,
    mean_mode: str = None,
    **kwargs,
):
    """
    CMVE 模式下的 InternVLModel.forward：
    - 图像特征获取和注入逻辑与官方一致
    - 语言模型 forward 被 monkey-patch 为 CMVE 版本
    """
    vision_outputs = self.get_image_features(pixel_values=pixel_values, **kwargs)
    image_features = vision_outputs.pooler_output.to(self.dtype)

    inputs_embeds = self.get_input_embeddings()(input_ids)
    special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
    inputs_embeds = inputs_embeds.masked_scatter(
        special_image_mask, image_features.contiguous().view(-1, image_features.shape[-1])
    )

    original_attn_forward = self.language_model.layers[0].attention.forward
    original_decoder_forward = self.language_model.layers[0].forward
    original_lang_forward = self.language_model.forward

    try:
        for layer in self.language_model.layers:
            layer.attention.forward = internvl_attn_forward.__get__(layer.attention, type(layer.attention))
            layer.forward = internvl_decoder_forward_new.__get__(layer, type(layer))
        self.language_model.forward = internvl_language_forward_new.__get__(self.language_model, type(self.language_model))

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            apply_cmve=apply_cmve,
            image_pos=image_pos,
            thres_mode=thres_mode,
            noise=noise,
            mean_mode=mean_mode,
            **kwargs,
        )
    finally:
        for layer in self.language_model.layers:
            layer.attention.forward = original_attn_forward
            layer.forward = original_decoder_forward
        self.language_model.forward = original_lang_forward

    return outputs


def internvl_forward_new(
    self: InternVLForConditionalGeneration,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    thres_mode: str = "high",
    noise: bool = False,
    mean_mode: str = "image",
    cmve: int = 1,
    **kwargs,
) -> InternVLCausalLMOutputWithPast:
    """
    顶层 CMVE forward，与 qwen2_vl_cmve.py 的模式对应。

    - cmve = 0: 完全走官方 InternVLForConditionalGeneration.forward 的逻辑（不改注意力层）
    - cmve = 1: 启用 CMVE，对语言模型内部注意力做修改
    """
    if cmve == 0:
        # 完全复用官方路径（不改注意力）
        outputs = internvl_language_forward_original(
            self.model,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return InternVLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    # cmve != 0: 启用 CMVE
    print("THIS IS CMVE FORWARD for InternVL")
    print("Using thres_mode:", thres_mode, " Using noise:", noise, " Using mean_mode:", mean_mode)

    # 构造 image_pos（简化：假设 image_token_id 对应的一段 token 连续）
    image_pos: List[List[dict]] = []
    for i in range(input_ids.shape[0]):
        sample_intervals: List[dict] = []
        image_token_indices = torch.where(input_ids[i] == self.config.image_token_id)[0]
        if len(image_token_indices) > 0:
            start_idx = image_token_indices[0].item()
            end_idx = image_token_indices[-1].item() + 1
            sample_intervals.append({"image_token_start": start_idx, "image_token_end": end_idx})
        image_pos.append(sample_intervals)

    original_model_forward = self.model.forward
    try:
        self.model.forward = internvl_base_forward_new.__get__(self.model, type(self.model))
        outputs_mod = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            apply_cmve=True,
            image_pos=image_pos,
            thres_mode=thres_mode,
            noise=noise,
            mean_mode=mean_mode,
            **kwargs,
        )
    finally:
        self.model.forward = original_model_forward

    hidden_states_cf = outputs_mod[0]
    logits_cf = self.lm_head(hidden_states_cf)

    return InternVLCausalLMOutputWithPast(
        loss=None,
        logits=logits_cf,
        past_key_values=outputs_mod.past_key_values,
        hidden_states=outputs_mod.hidden_states,
        attentions=outputs_mod.attentions,
        image_hidden_states=None,
    )