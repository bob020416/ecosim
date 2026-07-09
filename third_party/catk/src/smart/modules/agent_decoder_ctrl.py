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

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.modules.control import (
    AdditiveLayer,
    FiLMLayer,
    FutureEncoder,
    LatentEncoder,
    TextPromptEncoder,
    SinusoidalPositionalEncoding,
)
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)


class SMARTAgentDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        control_cfg: Optional[DictConfig] = None,
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )

        control_cfg = control_cfg or {}
        self._use_control = bool(control_cfg.get("use_control", False))
        if self._use_control:
            self._control_future_len = int(control_cfg.get("future_len", 16))
            self._control_feature_dim = int(control_cfg.get("feature_dim", 64))
            self._control_per_step = bool(control_cfg.get("per_step", False))
            self._control_remaining_window = bool(
                control_cfg.get("remaining_window", False)
            )
            self._control_condition_time_embed = bool(
                control_cfg.get("condition_time_embed", False)
            )
            sketch_mode = control_cfg.get("sketch_mode", None)
            if sketch_mode is None:
                self._control_sketch_mode = None
            else:
                self._control_sketch_mode = str(sketch_mode).lower()
                if self._control_sketch_mode not in {"dynamic", "static"}:
                    raise ValueError(
                        "control.sketch_mode must be 'dynamic' or 'static', "
                        f"got {sketch_mode}"
                    )
            self._control_window_len = int(control_cfg.get("window_len", 0))
            self._control_dropout = float(control_cfg.get("dropout", 0.0))
            self._control_no_control_prob = float(
                control_cfg.get("no_control_prob", 0.0)
            )
            self._control_zero_invalid = bool(control_cfg.get("zero_invalid", True))
            self._control_use_cross_agent_attn = bool(
                control_cfg.get("use_cross_agent_attn", False)
            )
            self._control_conditioning_mode = str(
                control_cfg.get("conditioning_mode", "sketch")
            ).lower()
            if self._control_conditioning_mode == "future":
                self._control_conditioning_mode = "sketch"
            if self._control_conditioning_mode not in {"sketch", "latent", "text"}:
                raise ValueError(
                    "control.conditioning_mode must be 'sketch', 'latent', or 'text', "
                    f"got {self._control_conditioning_mode}"
                )
            self._control_film_blocks = control_cfg.get(
                "film_blocks", list(range(num_layers))
            )
            self._control_fusion_strategy = str(
                control_cfg.get("fusion_strategy", "film")
            ).lower()
            if self._control_fusion_strategy not in {"film", "additive"}:
                raise ValueError(
                    "control.fusion_strategy must be 'film' or 'additive', "
                    f"got {self._control_fusion_strategy}"
                )
            self._control_film_hidden_dim = int(
                control_cfg.get("film_hidden_dim", 256)
            )
            self._control_film_dropout = float(
                control_cfg.get("film_dropout", 0.0)
            )
            self._control_film_init_mode = str(
                control_cfg.get("film_init_mode", "identity")
            ).lower()
            if self._control_film_init_mode not in {"identity", "random"}:
                raise ValueError(
                    "control.film_init_mode must be 'identity' or 'random', "
                    f"got {self._control_film_init_mode}"
                )
            self._control_film_identity_noise_std = float(
                control_cfg.get("film_identity_noise_std", 0.001)
            )
            if self._control_film_identity_noise_std < 0.0:
                raise ValueError(
                    "control.film_identity_noise_std must be >= 0.0, "
                    f"got {self._control_film_identity_noise_std}"
                )

            latent_use_cross_agent_attn = control_cfg.get(
                "latent_use_cross_agent_attn", None
            )
            if latent_use_cross_agent_attn is None:
                latent_use_cross_agent_attn = self._control_use_cross_agent_attn
            self._control_latent_agent_dim = int(
                control_cfg.get("latent_agent_dim", 512)
            )
            self._control_latent_scene_dim = int(
                control_cfg.get("latent_scene_dim", 0)
            )
            self._control_latent_hidden_dim = int(
                control_cfg.get("latent_hidden_dim", 256)
            )
            self._control_latent_n_layers = int(
                control_cfg.get("latent_n_layers", 3)
            )
            self._control_latent_dropout = float(
                control_cfg.get("latent_dropout", control_cfg.get("encoder_dropout", 0.0))
            )
            self._control_latent_cross_agent_heads = int(
                control_cfg.get("latent_cross_agent_heads", control_cfg.get("encoder_heads", 4))
            )

            if self._control_conditioning_mode == "sketch":
                self.control_encoder = FutureEncoder(
                    input_dim=3,
                    feature_dim=self._control_feature_dim,
                    agents_len=int(control_cfg.get("agents_len", 32)),
                    dropout=float(control_cfg.get("encoder_dropout", 0.0)),
                    n_layers=int(control_cfg.get("encoder_layers", 2)),
                    n_heads=int(control_cfg.get("encoder_heads", 4)),
                    d_ff=int(control_cfg.get("encoder_d_ff", 256)),
                    use_cross_agent_attn=self._control_use_cross_agent_attn,
                )
            elif self._control_conditioning_mode == "latent":
                self.control_encoder = LatentEncoder(
                    agent_latent_dim=self._control_latent_agent_dim,
                    scenario_latent_dim=self._control_latent_scene_dim,
                    feature_dim=self._control_feature_dim,
                    hidden_dim=self._control_latent_hidden_dim,
                    n_layers=self._control_latent_n_layers,
                    dropout=self._control_latent_dropout,
                    use_cross_agent_attn=bool(latent_use_cross_agent_attn),
                    cross_agent_heads=self._control_latent_cross_agent_heads,
                )
            else:
                self._control_text_max_length = int(
                    control_cfg.get("text_max_length", 128)
                )
                self._control_text_freeze_backbone = bool(
                    control_cfg.get("text_freeze_backbone", True)
                )
                self._control_text_use_mean_pool = bool(
                    control_cfg.get("text_use_mean_pool", True)
                )
                self._control_text_model_path = str(
                    control_cfg.get("text_model_path", "ckpt")
                )
                self._control_text_lora_rank = int(
                    control_cfg.get("text_lora_rank", 0)
                )
                self._control_text_lora_alpha = float(
                    control_cfg.get("text_lora_alpha", 16.0)
                )
                self._control_text_lora_dropout = float(
                    control_cfg.get("text_lora_dropout", 0.0)
                )
                self._control_text_lora_layers = int(
                    control_cfg.get("text_lora_layers", 0)
                )
                self._control_text_lora_targets = control_cfg.get(
                    "text_lora_target_modules", None
                )
                self.control_encoder = TextPromptEncoder(
                    model_name_or_path=self._control_text_model_path,
                    feature_dim=self._control_feature_dim,
                    max_length=self._control_text_max_length,
                    dropout=float(control_cfg.get("encoder_dropout", 0.0)),
                    freeze_backbone=self._control_text_freeze_backbone,
                    use_mean_pool=self._control_text_use_mean_pool,
                    lora_rank=self._control_text_lora_rank,
                    lora_alpha=self._control_text_lora_alpha,
                    lora_dropout=self._control_text_lora_dropout,
                    lora_layers=self._control_text_lora_layers,
                    lora_target_modules=self._control_text_lora_targets,
                )

            self.film_layers = nn.ModuleDict()
            fusion_layer_cls = (
                FiLMLayer
                if self._control_fusion_strategy == "film"
                else AdditiveLayer
            )
            for block_idx in self._control_film_blocks:
                layer_kwargs = dict(
                    feature_dim=hidden_dim,
                    conditioning_dim=self._control_feature_dim,
                    hidden_dim=self._control_film_hidden_dim,
                    dropout=self._control_film_dropout,
                )
                if self._control_fusion_strategy == "film":
                    layer_kwargs["init_mode"] = self._control_film_init_mode
                    layer_kwargs[
                        "identity_noise_std"
                    ] = self._control_film_identity_noise_std
                self.film_layers[f"block_{block_idx}"] = fusion_layer_cls(
                    **layer_kwargs
                )

            self._control_time_embed = None
            if self._control_condition_time_embed:
                max_len = max(self.num_future_steps, self._control_future_len, 100)
                self._control_time_embed = SinusoidalPositionalEncoding(
                    self._control_feature_dim, max_len=max_len
                )

        self.apply(weight_init)
        if self._use_control:
            # Preserve configured control fusion init after global weight init.
            for layer in self.film_layers.values():
                if hasattr(layer, "_init_weights"):
                    layer._init_weights()
            self.control_encoder._init_weights()

    def apply_base_lora(self, cfg: Dict) -> int:
        rank = int(cfg.get("rank", 0))
        if rank <= 0:
            return 0
        alpha = float(cfg.get("alpha", 16.0))
        dropout = float(cfg.get("dropout", 0.0))
        targets = cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out"])
        freeze_base = bool(cfg.get("freeze_base", True))
        unfreeze_control = bool(cfg.get("unfreeze_control", True))
        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False
        replaced = 0
        for layer in self.t_attn_layers:
            replaced += layer.apply_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=targets,
                freeze_base=freeze_base,
            )
        for layer in self.pt2a_attn_layers:
            replaced += layer.apply_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=targets,
                freeze_base=freeze_base,
            )
        for layer in self.a2a_attn_layers:
            replaced += layer.apply_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=targets,
                freeze_base=freeze_base,
            )
        if unfreeze_control and self._use_control:
            for p in self.control_encoder.parameters():
                p.requires_grad = True
            for p in self.film_layers.parameters():
                p.requires_grad = True
        return replaced

    def _control_action_start_10hz(self) -> int:
        return max(int(self.num_historical_steps) - 1, 0)

    def _control_action_start_2hz(self) -> int:
        start_10hz = self._control_action_start_10hz()
        if self.shift <= 0:
            return 0
        start_2hz = start_10hz // self.shift
        return max(start_2hz - 1, 0)

    def _slice_control_valid(self, valid: torch.Tensor, target_len: int) -> torch.Tensor:
        if target_len <= 0:
            return valid[:, :0]
        start_idx = self._control_action_start_2hz()
        if valid.shape[1] > start_idx:
            valid = valid[:, start_idx:]
        else:
            valid = valid[:, :0]
        if valid.shape[1] >= target_len:
            return valid[:, :target_len]
        pad_len = target_len - valid.shape[1]
        pad = torch.zeros(
            valid.shape[0], pad_len, device=valid.device, dtype=valid.dtype
        )
        return torch.cat([valid, pad], dim=1)

    def _build_control_sketch(self, tokenized_agent: Dict[str, torch.Tensor]):
        pos = tokenized_agent["gt_pos_raw"]
        head = tokenized_agent["gt_head_raw"]
        valid = tokenized_agent.get("gt_valid_raw")
        if self._control_future_len > 0:
            start_idx = self._control_action_start_2hz()
            target_len = self._control_future_len + 1
            end_idx = start_idx + target_len
            if pos.shape[1] >= end_idx:
                pos = pos[:, start_idx:end_idx]
                head = head[:, start_idx:end_idx]
                if valid is not None:
                    valid = valid[:, start_idx:end_idx]
            elif pos.shape[1] >= self._control_future_len:
                pos = pos[:, -self._control_future_len :]
                head = head[:, -self._control_future_len :]
                if valid is not None:
                    valid = valid[:, -self._control_future_len :]

        sketch = torch.cat([pos, head.unsqueeze(-1)], dim=-1)
        if self._control_zero_invalid and valid is not None:
            sketch = sketch * valid.unsqueeze(-1).float()
        return sketch

    def _encode_control_remaining(
        self,
        control_sketch: torch.Tensor,
    ):
        if control_sketch.dim() != 4:
            raise ValueError(
                f"control_sketch must be [B,A,T,3] for remaining window, got {control_sketch.shape}"
            )
        if control_sketch.shape[0] != 1:
            raise ValueError(
                "control_sketch batch dim must be 1 for remaining window; "
                f"got {control_sketch.shape[0]}"
            )
        _, _, total_len, _ = control_sketch.shape
        per_step_feats = []
        step_count = total_len
        if self._control_future_len > 0 and total_len > self._control_future_len:
            step_count = self._control_future_len
        for t in range(step_count):
            segment = control_sketch[:, :, t:, :]
            if self._control_window_len > 0:
                segment = segment[:, :, : self._control_window_len, :]
            step_feat = self.control_encoder(segment, return_sequence=False)
            per_step_feats.append(step_feat.squeeze(0))
        return torch.stack(per_step_feats, dim=1)

    def _encode_control_latents(
        self,
        control_latents: torch.Tensor,
        control_scene_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if control_latents.dim() == 4:
            if control_latents.shape[0] != 1:
                raise ValueError(
                    "control_latents batch dim must be 1; "
                    f"got {control_latents.shape[0]}"
                )
            control_latents = control_latents.squeeze(0)

        if control_latents.dim() == 2:
            if self._control_per_step:
                target_len = self._control_future_len if self._control_future_len > 0 else 1
                control_latents = control_latents[:, None, :].expand(-1, target_len, -1)
            else:
                control_feat = self.control_encoder(
                    control_latents.unsqueeze(0), control_scene_latents
                ).squeeze(0)
                return control_feat
        elif control_latents.dim() == 3:
            if not self._control_per_step:
                control_latents = control_latents[:, -1, :]
                control_feat = self.control_encoder(
                    control_latents.unsqueeze(0), control_scene_latents
                ).squeeze(0)
                return control_feat
        else:
            raise ValueError(
                f"control_latents must be [A,Za], [A,T,Za], or [1,A,T,Za], got {control_latents.shape}"
            )

        if control_scene_latents is not None:
            if control_scene_latents.dim() == 1:
                control_scene_latents = control_scene_latents.unsqueeze(0).expand(
                    control_latents.shape[1], -1
                )
            elif control_scene_latents.dim() == 2:
                if control_scene_latents.shape[0] == 1:
                    control_scene_latents = control_scene_latents.expand(
                        control_latents.shape[1], -1
                    )
                elif control_scene_latents.shape[0] != control_latents.shape[1]:
                    raise ValueError(
                        "control_scene_latents must be [T,Zs] for per-step latent conditioning, "
                        f"got {control_scene_latents.shape}"
                    )
            elif control_scene_latents.dim() == 3:
                if control_scene_latents.shape[0] == control_latents.shape[0]:
                    control_scene_latents = control_scene_latents.permute(1, 0, 2)
                elif control_scene_latents.shape[0] != control_latents.shape[1]:
                    raise ValueError(
                        "control_scene_latents must be [T,A,Zs] or [A,T,Zs], "
                        f"got {control_scene_latents.shape}"
                    )
            else:
                raise ValueError(
                    f"control_scene_latents must be [Zs], [T,Zs], or [A,T,Zs], got {control_scene_latents.shape}"
                )

        step_latents = control_latents.permute(1, 0, 2)  # [T,A,Za]
        control_feat = self.control_encoder(step_latents, control_scene_latents)
        return control_feat.permute(1, 0, 2)  # [A,T,F]

    def _encode_control_text(
        self,
        control_text: list,
        control_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # control_text is either a flat list of strings (per-agent) or
        # a list of per-scenario lists returned by the DataLoader.
        if not isinstance(control_text, list):
            raise ValueError("control_text must be a list of strings per agent")
        if control_text and isinstance(control_text[0], list):
            flat = []
            for group in control_text:
                if not isinstance(group, list):
                    raise ValueError("control_text nested list must contain lists per scenario")
                flat.extend(group)
            control_text = flat

        if control_text_mask is not None and len(control_text) != control_text_mask.shape[0]:
            raise ValueError(
                f"control_text length {len(control_text)} != text_prompt_mask length {control_text_mask.shape[0]}"
            )

        control_feat = self.control_encoder(control_text)
        if control_text_mask is not None:
            mask = control_text_mask.to(control_feat.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            control_feat = control_feat * mask.unsqueeze(-1)
        return control_feat

    def _align_control_seq(self, control_seq: torch.Tensor, target_len: int):
        if control_seq.shape[1] == target_len:
            return control_seq
        if control_seq.shape[1] < target_len:
            pad_len = target_len - control_seq.shape[1]
            if pad_len == 2:
                pad = torch.zeros(
                    control_seq.shape[0],
                    1,
                    control_seq.shape[2],
                    device=control_seq.device,
                    dtype=control_seq.dtype,
                )
                return torch.cat([pad, control_seq, pad], dim=1)
            pad = torch.zeros(
                control_seq.shape[0],
                pad_len,
                control_seq.shape[2],
                device=control_seq.device,
                dtype=control_seq.dtype,
            )
            return torch.cat([pad, control_seq], dim=1)
        return control_seq[:, -target_len:]

    def _add_control_time_embed(
        self,
        control_feat: torch.Tensor,
        target_len: int,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._control_condition_time_embed or self._control_time_embed is None:
            if control_feat.dim() == 3:
                return self._align_control_seq(control_feat, target_len)
            return control_feat
        if control_feat.dim() == 2:
            if step_idx is not None:
                step = min(step_idx, self._control_time_embed.pe.shape[0] - 1)
                return control_feat + self._control_time_embed.pe[step]
            control_feat = control_feat[:, None, :].expand(-1, target_len, -1)
        elif control_feat.dim() == 3:
            control_feat = self._align_control_seq(control_feat, target_len)
        else:
            raise ValueError(
                f"control_feat must be [A,F] or [A,T,F], got {control_feat.shape}"
            )
        control_feat = self._control_time_embed(
            control_feat.unsqueeze(0)
        ).squeeze(0)
        return control_feat

    def _build_control_mask(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        control_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if "gt_valid_raw" in tokenized_agent:
            valid = tokenized_agent["gt_valid_raw"]
        else:
            valid = tokenized_agent["valid_mask"]
        if self._control_future_len > 0:
            start_idx = self._control_action_start_2hz()
            end_idx = start_idx + self._control_future_len
            if valid.shape[1] >= end_idx:
                valid = valid[:, start_idx:end_idx]
            elif valid.shape[1] >= self._control_future_len:
                valid = valid[:, -self._control_future_len :]
        default_mask = valid.any(dim=1)

        if control_mask is None:
            return default_mask

        mask = control_mask
        if mask.dim() == 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() != 1:
            raise ValueError(
                f"control_mask must be [n_agent] or [1, n_agent], got {mask.shape}"
            )
        if mask.shape[0] != default_mask.shape[0]:
            raise ValueError(
                f"control_mask length {mask.shape[0]} != n_agent {default_mask.shape[0]}"
            )
        mask = mask.to(default_mask.device)
        if mask.dtype != torch.bool:
            mask = mask > 0
        return mask & default_mask

    def _encode_control(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        control_sketch: Optional[torch.Tensor] = None,
        control_mask: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        control_scene_latents: Optional[torch.Tensor] = None,
        control_text: Optional[list] = None,
        control_text_mask: Optional[torch.Tensor] = None,
    ):
        if not self._use_control:
            return None, None

        if self._control_conditioning_mode == "latent":
            if control_latents is None:
                raise ValueError(
                    "control_latents is required when conditioning_mode='latent'"
                )
            control_feat = self._encode_control_latents(
                control_latents, control_scene_latents
            )
        elif self._control_conditioning_mode == "text":
            if control_text is None:
                raise ValueError(
                    "control_text is required when conditioning_mode='text'"
                )
            control_feat = self._encode_control_text(control_text, control_text_mask)
        else:
            if control_sketch is None:
                control_sketch = self._build_control_sketch(tokenized_agent)
            if control_sketch.dim() == 3:
                control_sketch = control_sketch.unsqueeze(0)
            elif control_sketch.dim() != 4:
                raise ValueError(
                    f"control_sketch must be [A,T,3] or [B,A,T,3], got {control_sketch.shape}"
                )
            if control_sketch.shape[0] != 1:
                raise ValueError(
                    "control_sketch batch dim must be 1 for current control path; "
                    f"got {control_sketch.shape[0]}"
                )

            use_remaining_window = self._control_remaining_window
            if self._control_sketch_mode == "dynamic":
                use_remaining_window = True
            elif self._control_sketch_mode == "static":
                use_remaining_window = False

            if self._control_per_step and use_remaining_window:
                control_feat = self._encode_control_remaining(control_sketch)
            else:
                control_feat = self.control_encoder(
                    control_sketch, return_sequence=self._control_per_step
                ).squeeze(0)
                if (
                    self._control_per_step
                    and self._control_future_len > 0
                    and control_feat.shape[1] > self._control_future_len
                ):
                    control_feat = control_feat[:, : self._control_future_len]

            if self._control_zero_invalid and "gt_valid_raw" in tokenized_agent:
                valid = tokenized_agent["gt_valid_raw"]
                if self._control_per_step:
                    valid = self._slice_control_valid(valid, control_feat.shape[1])
                    control_feat = control_feat * valid.unsqueeze(-1)
                else:
                    target_len = (
                        self._control_future_len
                        if self._control_future_len > 0
                        else valid.shape[1]
                    )
                    valid = self._slice_control_valid(valid, target_len)
                    valid_any = valid.any(dim=1)
                    control_feat = control_feat * valid_any.unsqueeze(-1)

        control_mask = self._build_control_mask(tokenized_agent, control_mask)
        if self.training and self._control_no_control_prob > 0.0:
            if torch.rand((), device=control_feat.device) < self._control_no_control_prob:
                control_mask = torch.zeros_like(control_mask)
                return None, control_mask

        if self.training and self._control_dropout > 0.0:
            drop_mask = torch.rand(
                control_mask.shape[0], device=control_mask.device
            ) < self._control_dropout
            control_mask = control_mask & ~drop_mask

        if self._control_per_step:
            control_feat = control_feat * control_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            control_feat = control_feat * control_mask.unsqueeze(-1)

        if control_mask.sum() == 0:
            return None, control_mask

        return control_feat, control_mask

    def _apply_control_fusion(
        self,
        features: torch.Tensor,
        control_feat: Optional[torch.Tensor],
        block_idx: int,
        step_idx: Optional[int] = None,
        control_mask: Optional[torch.Tensor] = None,
    ):
        if control_feat is None:
            return features
        if control_mask is not None and control_mask.sum() == 0:
            return features
        key = f"block_{block_idx}"
        if key not in self.film_layers:
            return features
        film_layer = self.film_layers[key]

        if control_feat.dim() == 3:
            if features.dim() == 2:
                if step_idx is None:
                    raise ValueError("step_idx is required for per-step control.")
                cond = control_feat[:, step_idx]
                cond = self._add_control_time_embed(
                    cond, target_len=1, step_idx=step_idx
                )
                x = features.unsqueeze(0).unsqueeze(2)
                c = cond.unsqueeze(0)
                x = film_layer(x, c)
                out = x.squeeze(0).squeeze(1)
                if control_mask is None:
                    return out
                mask = control_mask.to(out.device).float()[:, None]
                return out * mask + features * (1.0 - mask)
            if features.dim() == 3:
                cond_seq = self._add_control_time_embed(
                    control_feat, target_len=features.shape[1]
                )
                x = features.unsqueeze(0)
                c = cond_seq.unsqueeze(0)
                x = film_layer(x, c)
                out = x.squeeze(0)
                if control_mask is None:
                    return out
                mask = control_mask.to(out.device).float()[:, None, None]
                return out * mask + features * (1.0 - mask)
            return features

        if features.dim() == 2:
            x = features.unsqueeze(0).unsqueeze(2)
            cond = self._add_control_time_embed(
                control_feat, target_len=1, step_idx=step_idx
            )
            c = cond.unsqueeze(0)
            x = film_layer(x, c)
            out = x.squeeze(0).squeeze(1)
            if control_mask is None:
                return out
            mask = control_mask.to(out.device).float()[:, None]
            return out * mask + features * (1.0 - mask)
        if features.dim() == 3:
            x = features.unsqueeze(0)
            cond_seq = self._add_control_time_embed(
                control_feat, target_len=features.shape[1]
            )
            c = cond_seq.unsqueeze(0)
            x = film_layer(x, c)
            out = x.squeeze(0)
            if control_mask is None:
                return out
            mask = control_mask.to(out.device).float()[:, None, None]
            return out * mask + features * (1.0 - mask)
        return features

    def _apply_control_film(
        self,
        features: torch.Tensor,
        control_feat: Optional[torch.Tensor],
        block_idx: int,
        step_idx: Optional[int] = None,
        control_mask: Optional[torch.Tensor] = None,
    ):
        # Backward-compatible alias.
        return self._apply_control_fusion(
            features=features,
            control_feat=control_feat,
            block_idx=block_idx,
            step_idx=step_idx,
            control_mask=control_mask,
        )

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token_veh,  # [n_token, 8]
        trajectory_token_ped,  # [n_token, 8]
        trajectory_token_cyc,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        emb_dtype = agent_token_emb_veh.dtype
        agent_token_emb_veh = agent_token_emb_veh.to(emb_dtype)
        agent_token_emb_ped = agent_token_emb_ped.to(emb_dtype)
        agent_token_emb_cyc = agent_token_emb_cyc.to(emb_dtype)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=emb_dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim).to(emb_dtype)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    def build_temporal_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2],
        mask,  # [n_agent, n_step]
        inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        batch_s,  # [n_agent*n_step]
        mask,  # [n_agent, n_step]
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl*n_step]
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        control_sketch: Optional[torch.Tensor] = None,
        control_mask: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        control_scene_latents: Optional[torch.Tensor] = None,
        control_text: Optional[list] = None,
        control_text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape
        control_feat, control_mask = self._encode_control(
            tokenized_agent,
            control_sketch,
            control_mask=control_mask,
            control_latents=control_latents,
            control_scene_latents=control_scene_latents,
            control_text=control_text,
            control_text_mask=control_text_mask,
        )

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_pl*n_step]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],  # [n_pl, 2]
            orient_pl=map_feature["orientation"],  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s,  # [n_agent*n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
        )

        # ! attention layers
        # [n_step*n_pl, hidden_dim]
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
            feat_a = self._apply_control_fusion(
                feat_a, control_feat, i, control_mask=control_mask
            )

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a)

        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
        control_sketch: Optional[torch.Tensor] = None,
        control_mask: Optional[torch.Tensor] = None,
        control_latents: Optional[torch.Tensor] = None,
        control_scene_latents: Optional[torch.Tensor] = None,
        control_text: Optional[list] = None,
        control_text_mask: Optional[torch.Tensor] = None,
        cfg_guidance_enabled: bool = False,
        cfg_guidance_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()
        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )
        control_feat, control_mask = self._encode_control(
            tokenized_agent,
            control_sketch,
            control_mask=control_mask,
            control_latents=control_latents,
            control_scene_latents=control_scene_latents,
            control_text=control_text,
            control_text_mask=control_text_mask,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )

        pred_valid = tokenized_agent["valid_mask"].clone()
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}
        cfg_guidance_scale = float(cfg_guidance_scale)
        use_cfg_mixing = bool(
            cfg_guidance_enabled and abs(cfg_guidance_scale - 1.0) > 1e-8
        )
        base_feat_a_t_dict = {} if use_cfg_mixing else None
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                batch_s=batch_s,  # [n_agent*hist_step]
                batch_pl=batch_pl,  # [n_pl*hist_step]
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            # ! attention layers
            for i in range(self.num_layers):
                if use_cfg_mixing:
                    _base_feat_temporal = (
                        feat_a if i == 0 else base_feat_a_t_dict[i]
                    )
                    if t == 0:
                        _base_feat_temporal = self.t_attn_layers[i](
                            _base_feat_temporal.flatten(0, 1), r_t, edge_index_t
                        ).view(n_agent, n_step, -1)
                        _base_feat_temporal = _base_feat_temporal.transpose(
                            0, 1
                        ).flatten(0, 1)
                        _base_feat_map = (
                            map_feature["pt_token"]
                            .unsqueeze(0)
                            .expand(hist_step, -1, -1)
                            .flatten(0, 1)
                        )
                        _base_feat_temporal = self.pt2a_attn_layers[i](
                            (_base_feat_map, _base_feat_temporal),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                        _base_feat_temporal = self.a2a_attn_layers[i](
                            _base_feat_temporal, r_a2a, edge_index_a2a
                        )
                        _base_feat_temporal = _base_feat_temporal.view(
                            n_step, n_agent, -1
                        ).transpose(0, 1)
                        base_feat_a_now = _base_feat_temporal[:, -1]
                        if i + 1 < self.num_layers:
                            base_feat_a_t_dict[i + 1] = _base_feat_temporal
                    else:
                        base_feat_a_now = self.t_attn_layers[i](
                            (
                                _base_feat_temporal.flatten(0, 1),
                                _base_feat_temporal[:, -1],
                            ),
                            r_t,
                            edge_index_t,
                        )
                        base_feat_a_now = self.pt2a_attn_layers[i](
                            (map_feature["pt_token"], base_feat_a_now),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                        base_feat_a_now = self.a2a_attn_layers[i](
                            base_feat_a_now, r_a2a, edge_index_a2a
                        )
                        if i + 1 < self.num_layers:
                            base_feat_a_t_dict[i + 1] = torch.cat(
                                (
                                    base_feat_a_t_dict[i + 1],
                                    base_feat_a_now.unsqueeze(1),
                                ),
                                dim=1,
                            )

                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    # [hist_step*n_pl, hidden_dim]
                    _feat_map = (
                        map_feature["pt_token"]
                        .unsqueeze(0)
                        .expand(hist_step, -1, -1)
                        .flatten(0, 1)
                    )

                    _feat_temporal = self.pt2a_attn_layers[i](
                        (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                    )
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]
                    feat_a_now = self._apply_control_fusion(
                        feat_a_now,
                        control_feat,
                        i,
                        step_idx=t,
                        control_mask=control_mask,
                    )

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )
                    # * give same results as below, but more efficient
                    # feat_a_now = self.t_attn_layers[i](
                    #     _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    # ).view(n_agent, n_step, -1)[:, -1]

                    feat_a_now = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], feat_a_now), r_pl2a, edge_index_pl2a
                    )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )
                    feat_a_now = self._apply_control_fusion(
                        feat_a_now,
                        control_feat,
                        i,
                        step_idx=t,
                        control_mask=control_mask,
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            control_next_token_logits = self.token_predict_head(feat_a_now)
            if use_cfg_mixing:
                base_next_token_logits = self.token_predict_head(base_feat_a_now)
                next_token_logits = base_next_token_logits + cfg_guidance_scale * (
                    control_next_token_logits - base_next_token_logits
                )
            else:
                next_token_logits = control_next_token_logits
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            next_token_idx, next_token_traj_all = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
        }

        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])

        return out_dict
