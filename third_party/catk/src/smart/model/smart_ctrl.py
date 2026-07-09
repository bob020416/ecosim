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

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.smart.modules.control import TextPromptEncoder
from src.smart.modules.smart_decoder_ctrl import SMARTDecoderCtrl
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import compute_batch_boundaries, freeze_rollouts_after_invalid, truncate_batch_by_boundary
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.map_boundary_cfg = model_config.get("map_boundary", {})
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.control_cfg = model_config.get("control", {})
        self._control_mask_from_train = bool(
            self.control_cfg.get("mask_from_train", False)
        )
        self.encoder = SMARTDecoderCtrl(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
            control_cfg=self.control_cfg,
            base_lora_cfg=model_config.get("base_lora", {}),
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)
        if self.control_cfg.get("freeze_base", False):
            self._freeze_base_for_control()

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    def _apply_map_boundary_to_rollouts(self, data, pred_traj, pred_head):
        if not bool(self.map_boundary_cfg.get("enable", False)):
            return pred_traj, pred_head
        boundaries = compute_batch_boundaries(
            data["map_save"]["traj_pos"],
            data["pt_token"]["batch"],
            data.num_graphs,
            margin=float(self.map_boundary_cfg.get("margin", 3.0)),
        )
        n_agent, n_rollout, n_step, _ = pred_traj.shape
        batch = data["agent"]["batch"]
        batch_rollout = batch.unsqueeze(1).expand(-1, n_rollout).reshape(-1)
        valid_mask = torch.ones(
            (n_agent * n_rollout, n_step),
            dtype=torch.bool,
            device=pred_traj.device,
        )
        valid_mask = truncate_batch_by_boundary(
            valid_mask,
            pred_traj.reshape(n_agent * n_rollout, n_step, 2),
            batch_rollout,
            boundaries,
            start_step=0,
        ).view(n_agent, n_rollout, n_step)
        pred_traj, pred_head = freeze_rollouts_after_invalid(valid_mask, pred_traj, pred_head)
        return pred_traj, pred_head

    def _zero_loss_with_grad(self) -> torch.Tensor:
        loss = None
        for p in self.parameters():
            if not p.requires_grad:
                continue
            term = p.sum() * 0.0
            loss = term if loss is None else (loss + term)
        if loss is None:
            return torch.zeros((), device=self.device, requires_grad=True)
        return loss

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        control_mask = None
        if self._control_mask_from_train and "train_mask" in data["agent"]:
            control_mask = data["agent"]["train_mask"]
        cond_mode = str(self.control_cfg.get("conditioning_mode", "sketch"))
        if cond_mode == "future":
            cond_mode = "sketch"
        if cond_mode == "latent" and "latent_mask" in data["agent"]:
            latent_mask = data["agent"]["latent_mask"]
            control_mask = (
                latent_mask
                if control_mask is None
                else (control_mask & latent_mask)
            )
        control_latents = data["agent"].get("latent", None)
        control_scene_latents = data.get("latent_scene", None)
        control_text = data["agent"].get("text_prompt", None)
        control_text_mask = data["agent"].get("text_prompt_mask", None)
        if cond_mode == "text" and control_text_mask is not None:
            if control_mask is None:
                control_mask = control_text_mask
            else:
                control_mask = control_mask & control_text_mask
        try:
            if self.training_rollout_sampling.num_k <= 0:
                pred = self.encoder(
                    tokenized_map,
                    tokenized_agent,
                    control_mask=control_mask,
                    control_latents=control_latents,
                    control_scene_latents=control_scene_latents,
                    control_text=control_text,
                    control_text_mask=control_text_mask,
                )
            else:
                pred = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    sampling_scheme=self.training_rollout_sampling,
                    control_mask=control_mask,
                    control_latents=control_latents,
                    control_scene_latents=control_scene_latents,
                    control_text=control_text,
                    control_text_mask=control_text_mask,
                )
        except ValueError as e:
            # Skip batches that produce NaN logits in categorical sampling.
            if "Categorical" in str(e) and "logits" in str(e):
                self.log("train/skip_nan", 1, on_step=True, batch_size=1)
                return self._zero_loss_with_grad()
            raise

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
        if (not torch.is_tensor(loss)) or (not loss.requires_grad):
            self.log("train/skip_no_grad", 1, on_step=True, batch_size=1)
            return self._zero_loss_with_grad()
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        control_mask = None
        if self._control_mask_from_train and "train_mask" in data["agent"]:
            control_mask = data["agent"]["train_mask"]
        cond_mode = str(self.control_cfg.get("conditioning_mode", "sketch"))
        if cond_mode == "future":
            cond_mode = "sketch"
        if cond_mode == "latent" and "latent_mask" in data["agent"]:
            latent_mask = data["agent"]["latent_mask"]
            control_mask = (
                latent_mask
                if control_mask is None
                else (control_mask & latent_mask)
            )
        control_latents = data["agent"].get("latent", None)
        control_scene_latents = data.get("latent_scene", None)
        control_text = data["agent"].get("text_prompt", None)
        control_text_mask = data["agent"].get("text_prompt_mask", None)
        if cond_mode == "text" and control_text_mask is not None:
            if control_mask is None:
                control_mask = control_text_mask
            else:
                control_mask = control_mask & control_text_mask

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(
                tokenized_map,
                tokenized_agent,
                control_mask=control_mask,
                control_latents=control_latents,
                control_scene_latents=control_scene_latents,
                control_text=control_text,
                control_text_mask=control_text_mask,
            )
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    self.validation_rollout_sampling,
                    control_mask=control_mask,
                    control_latents=control_latents,
                    control_scene_latents=control_scene_latents,
                    control_text=control_text,
                    control_text_mask=control_text_mask,
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
            pred_traj, pred_head = self._apply_map_boundary_to_rollouts(
                data, pred_traj, pred_head
            )

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    max_vis = min(
                        self.n_vis_scenario,
                        len(scenario_rollouts),
                        len(data["tfrecord_path"]),
                    )
                    for _i_sc in range(max_vis):
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        for _path in _vis.video_paths:
                            loggers = []
                            if getattr(self, "loggers", None):
                                loggers = self.loggers
                            elif self.logger is not None:
                                loggers = [self.logger]
                            for logger in loggers:
                                if hasattr(logger, "log_video"):
                                    logger.log_video(
                                        "/".join(_path.split("/")[-3:]), [_path]
                                    )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                realism = epoch_wosac_metrics.get(
                    "val_closed/wosac/realism_meta_metric"
                )
                min_ade = epoch_wosac_metrics.get("val_closed/wosac/min_ade")
                if realism is not None:
                    self.log(
                        "val_closed/wosac/realism_meta_metric",
                        realism,
                        logger=self.global_rank == 0,
                        on_step=False,
                        on_epoch=True,
                    )
                if min_ade is not None:
                    self.log(
                        "val_closed/wosac/min_ade",
                        min_ade,
                        logger=self.global_rank == 0,
                        on_step=False,
                        on_epoch=True,
                    )
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    if realism is not None:
                        epoch_wosac_metrics.pop(
                            "val_closed/wosac/realism_meta_metric", None
                        )
                    if min_ade is not None:
                        epoch_wosac_metrics.pop("val_closed/wosac/min_ade", None)
                    self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def _freeze_base_for_control(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Always keep LoRA parameters trainable (base or control).
        for name, p in self.encoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

        agent_encoder = self.encoder.agent_encoder
        if hasattr(agent_encoder, "control_encoder"):
            control_encoder = agent_encoder.control_encoder
            if isinstance(control_encoder, TextPromptEncoder):
                text_freeze_backbone = bool(
                    self.control_cfg.get("text_freeze_backbone", True)
                )
                for name, p in control_encoder.named_parameters():
                    if name.startswith("model.pooler"):
                        p.requires_grad = False
                    elif (
                        name.startswith("model.")
                        and text_freeze_backbone
                        and ".lora_" not in name
                    ):
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            else:
                for p in control_encoder.parameters():
                    p.requires_grad = True
        if hasattr(agent_encoder, "film_layers"):
            for p in agent_encoder.film_layers.parameters():
                p.requires_grad = True

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        control_text = data["agent"].get("text_prompt", None)
        control_text_mask = data["agent"].get("text_prompt_mask", None)
        control_mask = None
        cond_mode = str(self.control_cfg.get("conditioning_mode", "sketch"))
        if cond_mode == "future":
            cond_mode = "sketch"
        if cond_mode == "text" and control_text_mask is not None:
            control_mask = control_text_mask

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                self.validation_rollout_sampling,
                control_mask=control_mask,
                control_text=control_text,
                control_text_mask=control_text_mask,
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
        pred_traj, pred_head = self._apply_map_boundary_to_rollouts(
            data, pred_traj, pred_head
        )

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
