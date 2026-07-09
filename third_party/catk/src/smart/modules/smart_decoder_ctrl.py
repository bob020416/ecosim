# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder_ctrl import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoderCtrl(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        control_cfg: Optional[DictConfig] = None,
        base_lora_cfg: Optional[DictConfig] = None,
    ) -> None:
        super(SMARTDecoderCtrl, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            control_cfg=control_cfg,
        )
        self.base_lora_cfg = base_lora_cfg or {}
        if bool(self.base_lora_cfg.get("enable", False)):
            self.map_encoder.apply_base_lora(self.base_lora_cfg)
            self.agent_encoder.apply_base_lora(self.base_lora_cfg)

    def forward(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        control_sketch: Optional[Tensor] = None,
        control_mask: Optional[Tensor] = None,
        control_latents: Optional[Tensor] = None,
        control_scene_latents: Optional[Tensor] = None,
        control_text: Optional[list] = None,
        control_text_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder(
            tokenized_agent,
            map_feature,
            control_sketch=control_sketch,
            control_mask=control_mask,
            control_latents=control_latents,
            control_scene_latents=control_scene_latents,
            control_text=control_text,
            control_text_mask=control_text_mask,
        )
        return pred_dict

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
        control_sketch: Optional[Tensor] = None,
        control_mask: Optional[Tensor] = None,
        control_latents: Optional[Tensor] = None,
        control_scene_latents: Optional[Tensor] = None,
        control_text: Optional[list] = None,
        control_text_mask: Optional[Tensor] = None,
        cfg_guidance_enabled: bool = False,
        cfg_guidance_scale: float = 1.0,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        pred_dict = self.agent_encoder.inference(
            tokenized_agent,
            map_feature,
            sampling_scheme,
            control_sketch=control_sketch,
            control_mask=control_mask,
            control_latents=control_latents,
            control_scene_latents=control_scene_latents,
            control_text=control_text,
            control_text_mask=control_text_mask,
            cfg_guidance_enabled=cfg_guidance_enabled,
            cfg_guidance_scale=cfg_guidance_scale,
        )
        return pred_dict
