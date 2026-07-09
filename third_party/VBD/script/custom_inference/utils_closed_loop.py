# Ctrl VBD Closed-loop Testing with custom sketch and Video Visualization
# This is the CLOSED-LOOP version with receding horizon control
import argparse
import csv
import glob
import math
import os
import pickle
import time
from typing import Callable, Dict, List, Tuple

import jax
import matplotlib.transforms as transforms
import mediapy
import numpy as np
import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from PIL import Image

from vbd.data.dataset import WaymaxDataset
from vbd.model.utils import set_seed
from vbd.sim_agent.sim_actor import VBDTest

# Force CPU execution for TF/JAX dependencies (mirrors training behaviour)
tf.config.set_visible_devices([], 'GPU')
jax.config.update('jax_platform_name', 'cpu')

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
N_SIM_AGENTS = 32
VIEW_RANGE = 75           # Camera view range (meters)
COORD_LIMIT = 100_000      # Filter extreme coordinates when computing metrics
SKETCH_COLOR = '#D500F9'   # Bright magenta for custom sketch overlay

# Helper dataset used to generate anchors identical to the training pipeline
train_dataset = WaymaxDataset(
    data_dir=None,
    anchor_path='./vbd/data/cluster_64_center_dict.pkl',
)

from vbd.model.vbd_ctrl import VBD as ControlNetModule
from vbd.sim_agent.utils import duplicate_batch, torch_dict_to_numpy, stack_dict
from vbd.sim_agent.safety_filter import select_safest_trajectory
from tqdm import tqdm

SketchFn = Callable[[dict[str, torch.Tensor]], Tuple[np.ndarray | None, List[int]]]

# This class is custom for inference with ControlNet VBD we trained in finetune.py
class VBDCtrlTest(ControlNetModule):
    def __init__(
        self,
        cfg: dict,
        early_stop: int = 0,
        skip: int = 1,
        reward_func=None,
        guidance_iter: int = 5,
        guidance_end: int = 1,
        guidance_start: int = 99,
        gradient_scale: float = 1.0,
        scale_grad_by_std: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.early_stop = early_stop
        self.skip = skip
        self.reward_func = reward_func
        self.guidance_iter = guidance_iter
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.gradient_scale = gradient_scale
        self.scale_grad_by_std = scale_grad_by_std
        self.guidance_func = None

    def forward(self, batch, mode='sample_denoiser', **kwargs):
        """Forward pass wrapper for DataParallel support."""
        if mode == 'sample_denoiser':
            return self.sample_denoiser(batch, **kwargs)
        elif mode == 'inference_predictor':
            return self.inference_predictor(batch, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_agent_length(self, _agents_len: int) -> None:
        self._agents_len = _agents_len
        if self.predictor is not None:
            self.predictor.reset_agent_length(_agents_len)
        if self.denoiser is not None:
            self.denoiser.reset_agent_length(_agents_len)

    def inference_predictor(self, batch) -> dict:
        if self.predictor is None:
            raise RuntimeError("Predictor is not defined")
        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)
        return self.forward_predictor(encoder_outputs)

    def step_denoiser(
        self,
        x_t: torch.Tensor,
        c: dict,
        t: int,
        replan_idx: int | None = None,
        control_agents_future: torch.Tensor | None = None,
        control_agent_latents: torch.Tensor | None = None,
        control_scenario_latents: torch.Tensor | None = None,
        control_text_prompt=None,
        control_text_mask: torch.Tensor | None = None,
        control_mask: torch.Tensor | None = None,
    ):
        """
        Single diffusion step to obtain x_{t-1} given x_t and encoder outputs.
        """
        if self.denoiser is None:
            raise RuntimeError("Denoiser is not defined")

        # Conditioning only affects agents with valid mask entries; others remain baseline.
        denoiser_output = self.forward_denoiser(
            encoder_outputs=c,
            noised_actions_normalized=x_t,
            diffusion_step=t,
            agents_future=(
                control_agents_future[:, :self._agents_len]
                if control_agents_future is not None
                else None
            ),
            agent_latents=(
                control_agent_latents[:, :self._agents_len]
                if control_agent_latents is not None
                else None
            ),
            scenario_latents=(
                control_scenario_latents[:, :self._agents_len]
                if control_scenario_latents is not None
                else None
            ),
            text_prompt=control_text_prompt,
            text_prompt_mask=control_text_mask,
            replan_idx=replan_idx,
            control_mask=(
                control_mask[:, :self._agents_len]
                if control_mask is not None
                else None
            ),
        )

        x_0 = denoiser_output['denoised_actions_normalized']
        x_t_prev = self.noise_scheduler.step(
            model_output=x_0,
            timesteps=t,
            sample=x_t,
            prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
        )

        return denoiser_output, x_t_prev

    @torch.no_grad()
    def sample_denoiser(
        self,
        batch,
        num_samples: int = 1,
        x_t = None,
        use_tqdm: bool = True,
        fix_t: int = -1,
        calc_loss: bool = False,
        save_history: bool = False,
        replan_idx: int | None = None,
        **kwargs,
    ) -> dict:
        batch = self.batch_to_device(batch, self.device)

        conditioning_future = None
        conditioning_agent_latents = None
        conditioning_scenario_latents = None
        control_mask = None
        control_text_prompt = None
        control_text_mask = None

        # Handle text conditioning first (text model still carries agents_future in batch).
        if self._conditioning_mode == 'text':
            control_text_prompt = batch.get('text_prompt', None)
            control_text_mask = batch.get('text_prompt_mask', None)
            if control_text_prompt is not None:
                batch_size = batch['agents_history'].shape[0]
                if isinstance(control_text_prompt, (list, tuple)):
                    if control_text_prompt and isinstance(control_text_prompt[0], str):
                        control_text_prompt = [list(control_text_prompt) for _ in range(batch_size)]
                    else:
                        if len(control_text_prompt) == 1 and batch_size > 1:
                            control_text_prompt = [list(control_text_prompt[0]) for _ in range(batch_size)]
                        elif len(control_text_prompt) != batch_size:
                            if len(control_text_prompt) > batch_size:
                                control_text_prompt = [list(row) for row in control_text_prompt[:batch_size]]
                            else:
                                pad_row = list(control_text_prompt[0]) if control_text_prompt else []
                                control_text_prompt = [list(row) for row in control_text_prompt]
                                control_text_prompt.extend([list(pad_row) for _ in range(batch_size - len(control_text_prompt))])
            if isinstance(control_text_mask, torch.Tensor):
                if control_text_mask.dim() == 1:
                    control_text_mask = control_text_mask.unsqueeze(0)
                if control_text_mask.shape[0] == 1 and batch['agents_history'].shape[0] > 1:
                    control_text_mask = control_text_mask.repeat(batch['agents_history'].shape[0], 1)
                elif control_text_mask.shape[0] != batch['agents_history'].shape[0]:
                    if control_text_mask.shape[0] > batch['agents_history'].shape[0]:
                        control_text_mask = control_text_mask[: batch['agents_history'].shape[0]]
                    else:
                        pad = torch.zeros(
                            batch['agents_history'].shape[0] - control_text_mask.shape[0],
                            control_text_mask.shape[1],
                            device=control_text_mask.device,
                            dtype=control_text_mask.dtype,
                        )
                        control_text_mask = torch.cat([control_text_mask, pad], dim=0)
            if 'agents_interested' in batch:
                interested = batch['agents_interested'][:, :self._agents_len] > 0
                control_mask = interested.float()
                if isinstance(control_text_mask, torch.Tensor):
                    text_mask = control_text_mask
                    if text_mask.dim() == 1:
                        text_mask = text_mask.unsqueeze(0)
                    if text_mask.shape[0] == 1 and control_mask.shape[0] > 1:
                        text_mask = text_mask.repeat(control_mask.shape[0], 1)
                    # Align mask to control_mask length.
                    if text_mask.shape[1] != control_mask.shape[1]:
                        if text_mask.shape[1] > control_mask.shape[1]:
                            text_mask = text_mask[:, : control_mask.shape[1]]
                        else:
                            pad = torch.zeros(
                                text_mask.shape[0],
                                control_mask.shape[1] - text_mask.shape[1],
                                device=text_mask.device,
                                dtype=text_mask.dtype,
                            )
                            text_mask = torch.cat([text_mask, pad], dim=1)
                    if text_mask.dtype != torch.bool:
                        text_mask = text_mask > 0
                    control_mask = control_mask * text_mask.float()
        # Handle latent conditioning (agent_latents + scenario_latents)
        elif 'agent_latents' in batch and 'scenario_latents' in batch:
            conditioning_agent_latents = batch['agent_latents'][:, :self._agents_len]
            conditioning_scenario_latents = batch['scenario_latents'][:, :self._agents_len]
            # Agents are conditioned when they have any non-zero latent
            latent_non_zero = torch.ne(conditioning_agent_latents, 0.0).any(dim=-1)
            if 'agents_interested' in batch:
                interested = batch['agents_interested'][:, :self._agents_len] > 0
                latent_non_zero = latent_non_zero & interested
            control_mask = latent_non_zero.float()
        # Handle sketch conditioning (agents_future)
        elif 'agents_future' in batch:
            conditioning_future = batch['agents_future'][:, :self._agents_len]
            # Mirror training: agents are conditioned when they have any non-zero future state
            # and are of interest (if that signal is available).
            future_non_zero = torch.ne(conditioning_future, 0.0).any(dim=-1)
            has_condition = future_non_zero.any(dim=-1)
            if 'agents_interested' in batch:
                interested = batch['agents_interested'][:, :self._agents_len] > 0
                has_condition = has_condition & interested
            control_mask = has_condition.float()
        if calc_loss:
            agents_future = batch['agents_future'][:, :self._agents_len]
            agents_future_valid = torch.ne(agents_future.sum(-1), 0)
            agents_interested = batch['agents_interested'][:, :self._agents_len]
        
        t_enc_start = time.time()
        encoder_outputs = self.encoder(batch)


        if num_samples > 1:
            t_dup_start = time.time()
            encoder_outputs = duplicate_batch(encoder_outputs, num_samples)
            if conditioning_future is not None:
                conditioning_future = conditioning_future.repeat(num_samples, 1, 1, 1)
            if conditioning_agent_latents is not None:
                conditioning_agent_latents = conditioning_agent_latents.repeat(num_samples, 1, 1)
            if conditioning_scenario_latents is not None:
                conditioning_scenario_latents = conditioning_scenario_latents.repeat(num_samples, 1, 1)
            if control_mask is not None:
                control_mask = control_mask.repeat(num_samples, 1)
            if control_text_prompt is not None:
                expanded = []
                for row in control_text_prompt:
                    for _ in range(num_samples):
                        expanded.append(list(row))
                control_text_prompt = expanded
            if control_text_mask is not None:
                control_text_mask = control_text_mask.repeat_interleave(num_samples, dim=0)
      
        
        agents_history = encoder_outputs['agents']
    
        num_batch, num_agent = agents_history.shape[:2]
        num_step = self._future_len // self._action_len
        action_dim = 2
        diffusion_steps = list(reversed(range(self.early_stop, self.noise_scheduler.num_steps, self.skip)))
        x_t_history = []
        denoiser_output_history = []
        guide_history = []
        if x_t is None:
            x_t = torch.randn(num_batch, num_agent, num_step, action_dim, device=self.device)
        else:
            x_t = x_t.to(self.device)
        
        t_diff_start = time.time()
        iterator = tqdm(diffusion_steps, total=len(diffusion_steps), desc="Diffusion") if use_tqdm else diffusion_steps
        step_times = []
        for t in iterator:
            t_step_start = time.time()
            if save_history:
                x_t_history.append(x_t.detach().cpu().numpy())
            denoiser_outputs, x_t = self.step_denoiser(
                x_t=x_t,
                c=encoder_outputs,
                t=fix_t if fix_t >= 0 else t,
                replan_idx=replan_idx,
                control_agents_future=conditioning_future,
                control_agent_latents=conditioning_agent_latents,
                control_scenario_latents=conditioning_scenario_latents,
                control_text_prompt=control_text_prompt,
                control_text_mask=control_text_mask,
                control_mask=control_mask,
            )
            if calc_loss:
                denoised_trajs = denoiser_outputs['denoised_trajs']
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )
                denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                    denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
                )
                denoiser_outputs.update(
                    {
                        'state_loss_mean': state_loss_mean,
                        'yaw_loss_mean': yaw_loss_mean,
                        'denoise_ade': denoise_ade,
                        'denoise_fde': denoise_fde,
                    }
                )
            if save_history:
                denoiser_output_history.append(torch_dict_to_numpy(denoiser_outputs))
            step_times.append(time.time() - t_step_start)

        denoiser_outputs['history'] = None
        if save_history:
            denoiser_outputs['history'] = {
                'x_t_history': np.stack(x_t_history, axis=0),
                'denoiser_output_history': stack_dict(denoiser_output_history),
                'guide_history': stack_dict(guide_history),
            }
        return denoiser_outputs


# -----------------------------------------------------------------------------
# PKL utilities
# -----------------------------------------------------------------------------
def load_pkl_files(data_dir: str) -> List[str]:
    """Return a sorted list of PKL files inside *data_dir*."""
    pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
    print(f"Found {len(pkl_files)} pkl files in {data_dir}")
    return pkl_files


def load_pkl_data(pkl_file: str) -> dict:
    """Load a preprocessed scenario stored as a PKL file."""
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# Batch preparation
# -----------------------------------------------------------------------------
def prepare_model_batch(pkl_data: dict) -> dict:
    """Convert processed PKL data into a batch compatible with the model."""
    batch: dict[str, torch.Tensor] = {}

    agent_hist = pkl_data.get('agents_history')
    orig_agent_count = agent_hist.shape[0] if agent_hist is not None else 0
    n_agents = min(N_SIM_AGENTS, orig_agent_count if orig_agent_count else N_SIM_AGENTS)
    n_polylines = pkl_data.get('polylines', np.zeros((0,))).shape[0] if 'polylines' in pkl_data else 0
    n_traffic_lights = (
        pkl_data.get('traffic_light_points', np.zeros((0,))).shape[0]
        if 'traffic_light_points' in pkl_data
        else 0
    )

# ... (inside prepare_model_batch)
    keys = [
        'agents_history', 'agents_interested', 'agents_type', 'agents_future',
        'traffic_light_points', 'polylines', 'polylines_valid', 'relations',
        'agents_id'  # Added agents_id
    ]

    for key in keys:
        if key not in pkl_data:
            continue

        data = pkl_data[key]

        if key in {'agents_history', 'agents_interested', 'agents_type', 'agents_future', 'agents_id'}: # Added agents_id
            data = data[:n_agents]
# ... (rest of loop)


        elif key == 'relations':
            total = data.shape[0]
            if total != data.shape[1]:
                raise ValueError(f"relations tensor must be square, got {data.shape}")

            agent_indices = list(range(min(n_agents, total)))
            poly_start = orig_agent_count
            poly_end = min(poly_start + n_polylines, total)
            traffic_start = poly_end
            traffic_end = min(traffic_start + n_traffic_lights, total)

            keep_indices = agent_indices
            keep_indices += list(range(poly_start, poly_end))
            keep_indices += list(range(traffic_start, traffic_end))

            if not keep_indices:
                raise ValueError('Unable to derive indices for relations tensor slicing')

            data = data[np.ix_(keep_indices, keep_indices)]

        tensor = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        batch[key] = tensor

    if 'agents_type' in pkl_data:
        anchors_source = pkl_data['agents_type'][:n_agents]
        anchors = train_dataset._process(anchors_source)
        batch['anchors'] = torch.from_numpy(np.ascontiguousarray(anchors)).unsqueeze(0)

    if 'scenario_id' in pkl_data:
        batch['scenario_id'] = pkl_data['scenario_id']

    return batch


def override_future_with_forward_step(
    batch: dict[str, torch.Tensor],
    distance: float = 5.0,
    target_agents: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Overwrite selected agents with a forward sketch while leaving others untouched."""

    if 'agents_future' not in batch or 'agents_history' not in batch:
        return

    agents_future = batch['agents_future']  # [B, A, T, C]
    agents_history = batch['agents_history']  # [B, A, H, D]

    if agents_future.numel() == 0 or agents_history.numel() == 0:
        return

    batch_size, num_agents, future_len, future_dim = agents_future.shape

    # Filter the requested agent indices so we only touch valid entries.
    valid_indices = [idx for idx in target_agents if 0 <= idx < num_agents]
    if not valid_indices:
        return

    device = agents_future.device
    # Track which agents receive the synthetic sketch; used later to blank out others.
    selected_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
    selected_mask[:, valid_indices] = True

    current = agents_history[:, :, -1, :]  # latest history state per agent [B, A, D]
    # Offsets along the heading direction, spaced uniformly from 0 to `distance` metres.
    steps = torch.linspace(0.0, distance, future_len, device=device).view(1, 1, future_len)

    selected_current = current[:, valid_indices]  # state of the agents we override [B, |idx|, D]
    heading = selected_current[..., 2]
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)

    start_x = selected_current[..., 0].unsqueeze(-1)
    start_y = selected_current[..., 1].unsqueeze(-1)

    traj_x = start_x + cos_h * steps
    traj_y = start_y + sin_h * steps
    traj_yaw = heading.unsqueeze(-1).expand(-1, -1, future_len)

    # Inject the custom sketch (x, y, yaw) for each target agent.
    agents_future[:, valid_indices, :, 0] = traj_x
    agents_future[:, valid_indices, :, 1] = traj_y
    agents_future[:, valid_indices, :, 2] = traj_yaw

    # Clear any trailing channels (velocity, size, etc.) so no stale values leak downstream.
    if future_dim > 3:
        agents_future[:, valid_indices, :, 3:] = 0.0

    if 'agents_interested' in batch:
        uninterested_mask = batch['agents_interested'] <= 0
    else:
        uninterested_mask = current[..., :2].abs().sum(-1) <= 0

    if uninterested_mask.any():
        fill = torch.zeros_like(agents_future)  # align inference with training dropout
        # Only blank out uninterested agents that we did not explicitly target.
        mask = uninterested_mask & (~selected_mask)
        if mask.any():
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            agents_future[:] = torch.where(mask, fill, agents_future)

    # Finally, reset all non-target agents to zero so ControlNet/visualization only see the selected ones.
    non_target_mask = ~selected_mask
    if non_target_mask.any():
        fill = torch.zeros_like(agents_future)
        mask = non_target_mask.unsqueeze(-1).unsqueeze(-1)
        agents_future[:] = torch.where(mask, fill, agents_future)


def snapshot_conditioning(batch: dict[str, torch.Tensor]) -> Tuple[np.ndarray | None, List[int]]:
    """Return conditioning future and controlled agent ids from the batch."""

    agents_future_tensor = batch.get('agents_future')
    if agents_future_tensor is None or agents_future_tensor.numel() == 0:
        return None, []

    future_np = agents_future_tensor[0].detach().cpu().numpy().copy()
    controlled_ids = extract_controlled_agent_indices(future_np)
    return future_np, controlled_ids


def apply_forward_sketch(
    batch: dict[str, torch.Tensor],
    distance: float,
    target_agents: tuple[int, ...] = (0, 1, 2),
) -> Tuple[np.ndarray | None, List[int]]:
    """Apply forward sketch with the given distance and return the snapshot."""

    override_future_with_forward_step(
        batch,
        distance=distance,
        target_agents=target_agents,
    )
    return snapshot_conditioning(batch)




def apply_right_turn_sketch(
    batch: dict[str, torch.Tensor],
    forward_distance: float = 5.0,
    turn_distance: float = 5.0,
    turn_angle_deg: float = 60.0,
    target_agents: tuple[int, ...] = (0, 1, 2),
) -> Tuple[np.ndarray | None, List[int]]:
    """Sketch a forward segment followed by a smooth right turn for selected agents."""

    if 'agents_future' not in batch or 'agents_history' not in batch:
        return snapshot_conditioning(batch)

    agents_future = batch['agents_future']  # [B, A, T, D]
    agents_history = batch['agents_history']  # [B, A, H, D]

    if agents_future.numel() == 0 or agents_history.numel() == 0:
        return snapshot_conditioning(batch)

    batch_size, num_agents, future_len, future_dim = agents_future.shape
    valid_indices = [idx for idx in target_agents if 0 <= idx < num_agents]
    if not valid_indices:
        return snapshot_conditioning(batch)

    device = agents_future.device

    agents_future.zero_()

    total_distance = max(forward_distance + turn_distance, 1e-3)
    angle_rad = math.radians(turn_angle_deg)

    for agent_idx in valid_indices:
        start_state = agents_history[:, agent_idx, -1, :]  # [B, D]
        heading = start_state[:, 2]
        start_x = start_state[:, 0]
        start_y = start_state[:, 1]

        seg1_len = max(1, int(round(future_len * forward_distance / total_distance)))
        seg1_len = min(seg1_len, future_len)
        seg2_len = future_len - seg1_len

        dist1 = torch.linspace(0.0, forward_distance, seg1_len, device=device)
        cos_h = torch.cos(heading).unsqueeze(-1)
        sin_h = torch.sin(heading).unsqueeze(-1)

        traj_x_parts = []
        traj_y_parts = []
        yaw_parts = []

        x1 = start_x.unsqueeze(-1) + cos_h * dist1.unsqueeze(0)
        y1 = start_y.unsqueeze(-1) + sin_h * dist1.unsqueeze(0)
        yaw1 = heading.unsqueeze(-1).expand(-1, seg1_len)

        traj_x_parts.append(x1)
        traj_y_parts.append(y1)
        yaw_parts.append(yaw1)

        if seg2_len > 0:
            step_size = turn_distance / seg2_len
            turn_fracs = torch.linspace(1.0 / seg2_len, 1.0, seg2_len, device=device)
            headings_turn = heading.unsqueeze(-1) - angle_rad * turn_fracs.unsqueeze(0)

            dx = torch.cos(headings_turn) * step_size
            dy = torch.sin(headings_turn) * step_size

            offsets_x = torch.cumsum(dx, dim=1)
            offsets_y = torch.cumsum(dy, dim=1)

            last_x = x1[:, -1:]
            last_y = y1[:, -1:]

            x2 = last_x + offsets_x
            y2 = last_y + offsets_y
            yaw2 = headings_turn

            traj_x_parts.append(x2)
            traj_y_parts.append(y2)
            yaw_parts.append(yaw2)

        traj_x = torch.cat(traj_x_parts, dim=1)
        traj_y = torch.cat(traj_y_parts, dim=1)
        traj_yaw = torch.cat(yaw_parts, dim=1)

        agents_future[:, agent_idx, :, 0] = traj_x[:, :future_len]
        agents_future[:, agent_idx, :, 1] = traj_y[:, :future_len]
        if future_dim > 2:
            agents_future[:, agent_idx, :, 2] = traj_yaw[:, :future_len]
        if future_dim > 3:
            agents_future[:, agent_idx, :, 3:] = 0.0

    selected_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
    selected_mask[:, valid_indices] = True

    if 'agents_interested' in batch:
        uninterested_mask = batch['agents_interested'] <= 0
    else:
        current = agents_history[:, :, -1, :]
        uninterested_mask = current[..., :2].abs().sum(-1) <= 0

    if uninterested_mask.any():
        fill = torch.zeros_like(agents_future)
        mask = uninterested_mask & (~selected_mask)
        if mask.any():
            agents_future[:] = torch.where(mask.unsqueeze(-1).unsqueeze(-1), fill, agents_future)

    non_target_mask = ~selected_mask
    if non_target_mask.any():
        fill = torch.zeros_like(agents_future)
        agents_future[:] = torch.where(non_target_mask.unsqueeze(-1).unsqueeze(-1), fill, agents_future)

    return snapshot_conditioning(batch)


def extract_controlled_agent_indices(agents_future: np.ndarray | None) -> List[int]:
    """Return indices for agents with a conditioning sketch."""

    if agents_future is None or agents_future.ndim < 3:
        return []

    spatial = np.abs(agents_future[..., :2])

    if spatial.ndim == 3:  # [A, T, 2]
        magnitude = spatial.sum(axis=(1, 2))
        mask = magnitude > 1e-4
        return [int(idx) for idx, flagged in enumerate(mask) if flagged]

    if spatial.ndim == 4:  # [B, A, T, 2]
        magnitude = spatial.sum(axis=(2, 3))  # [B, A]
        mask = magnitude > 1e-4
        controlled: List[int] = []
        for agent_idx in range(mask.shape[1]):
            if mask[:, agent_idx].any():
                controlled.append(agent_idx)
        return controlled

    return []

# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------


def plot_controlled_trajectories(
    agents_history: np.ndarray,
    control_samples: np.ndarray,
    controlled_ids: List[int],
    polylines: np.ndarray | None,
    traffic_lights: np.ndarray | None,
    conditioning_future: np.ndarray | None,
    sketch_label: str,
    save_path: str,
) -> None:
    """Render multi-sample ControlNet trajectories plus sketches into a single image."""

    if control_samples.size == 0 or not controlled_ids:
        return

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8))

    if polylines is not None and polylines.size > 0:
        for line in polylines:
            if line[0, 0] == 0:
                continue
            valid = (
                (np.abs(line[:, 0]) < COORD_LIMIT)
                & (np.abs(line[:, 1]) < COORD_LIMIT)
            )
            if np.any(valid):
                ax.plot(line[valid][:, 0], line[valid][:, 1], '.', color='gray', alpha=0.3, markersize=1)

    if agents_history is not None and agents_history.size > 0:
        for hist in agents_history[:N_SIM_AGENTS]:
            valid = (
                (hist[:, 0] != 0)
                & (np.abs(hist[:, 0]) < COORD_LIMIT)
                & (np.abs(hist[:, 1]) < COORD_LIMIT)
            )
            if np.any(valid):
                ax.plot(hist[valid][:, 0], hist[valid][:, 1], '-', color='#4FC3F7', linewidth=0.5, alpha=0.4)

    cmap = plt.get_cmap('tab10')
    num_samples = control_samples.shape[0]

    for series_idx, agent_id in enumerate(controlled_ids):
        color = cmap(series_idx % 10)
        for sample in control_samples:
            if agent_id >= sample.shape[0]:
                continue
            traj = sample[agent_id]
            if traj.size == 0:
                continue
            valid = (
                (traj[:, 0] != 0)
                & (traj[:, 0] != -1)
                & (np.abs(traj[:, 0]) < COORD_LIMIT)
                & (np.abs(traj[:, 1]) < COORD_LIMIT)
            )
            if np.sum(valid) < 2:
                continue
            ax.plot(
                traj[valid][:, 0],
                traj[valid][:, 1],
                '-',
                color=color,
                alpha=0.15,
                linewidth=1,
            )

        if (
            conditioning_future is not None
            and agent_id < conditioning_future.shape[0]
        ):
            sketch = conditioning_future[agent_id]
            if sketch.ndim >= 2:
                valid = (
                    (sketch[:, 0] != 0)
                    & (sketch[:, 0] != -1)
                    & (np.abs(sketch[:, 0]) < COORD_LIMIT)
                    & (np.abs(sketch[:, 1]) < COORD_LIMIT)
                )
                if np.sum(valid) >= 2:
                    ax.plot(
                        sketch[valid][:, 0],
                        sketch[valid][:, 1],
                        '--',
                        color=SKETCH_COLOR,
                        alpha=0.9,
                        linewidth=2,
                        label=f'{sketch_label} (agent {agent_id})',
                    )

        ax.plot([], [], color=color, alpha=0.4, label=f'Agent {agent_id} samples (n={num_samples})')

    if traffic_lights is not None and traffic_lights.size > 0:
        # IMPROVED: Show traffic light color based on state
        # State mapping: 0=unknown, 1/4/7=red, 2/5/8=yellow, 3/6=green
        for tl in traffic_lights:
            if tl[0] == 0:
                continue
            if (
                np.abs(tl[0]) < COORD_LIMIT
                and np.abs(tl[1]) < COORD_LIMIT
            ):
                # Determine color based on state (if available)
                if tl.shape[0] >= 3:
                    state = int(tl[2])
                    if state in [1, 4, 7]:  # STOP states
                        color = '#FF5252'  # Red
                    elif state in [2, 5, 8]:  # CAUTION states
                        color = '#FFC107'  # Yellow/Amber
                    elif state in [3, 6]:  # GO states
                        color = '#4CAF50'  # Green
                    else:  # UNKNOWN
                        color = '#9E9E9E'  # Gray
                else:
                    color = '#FF5252'  # Default red
                ax.add_patch(plt.Circle((tl[0], tl[1]), 1.0, color=color, alpha=0.7))

    ax.set_title('Controlled Agent Trajectories (32 samples)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')
    ax.legend(loc='upper right')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def create_trajectory_video(
    pkl_data: dict,
    pred_trajs: List[Tuple[str, np.ndarray]],
    save_path: str | None,
    scenario_name: str,
    conditioning_futures: Dict[str, np.ndarray] | None = None,
    controlled_agent_ids: List[int] | None = None,
) -> List[np.ndarray]:
    """Render GT and one or more predictions in a modern dark-themed layout."""
    original_style = plt.rcParams['axes.prop_cycle']
    plt.style.use('dark_background')

    palette = {
        'map': '#424242',
        'gt_future': '#FFAB91',
        'pred_future': '#80CBC4',
        'pred_future_alt': '#B39DDB',
        'condition_future': '#FFE082',
        'ego_box': '#FFD54F',
        'others_box': '#F06292',
        'controlled_box': '#00E676',  # Bright green for controlled agents
    }

    agents_history = pkl_data['agents_history']
    agents_future = pkl_data['agents_future']
    polylines = pkl_data['polylines']
    traffic_lights = pkl_data['traffic_light_points']

    def make_future_tensor(pred_traj: np.ndarray) -> np.ndarray:
        future = -np.ones_like(agents_future)
        if pred_traj.shape[0] == 0:
            return future
        n_pred_agents = min(pred_traj.shape[0], N_SIM_AGENTS)
        n_pred_timesteps = min(pred_traj.shape[1], 80)
        valid_pred = pred_traj[:n_pred_agents, :n_pred_timesteps, :3]
        coord_mask = (
            (np.abs(valid_pred[:, :, 0]) < COORD_LIMIT)
            & (np.abs(valid_pred[:, :, 1]) < COORD_LIMIT)
        )
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, :3] = valid_pred
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, 0][~coord_mask] = -1
        future[:n_pred_agents, 1 : n_pred_timesteps + 1, 1][~coord_mask] = -1

        if n_pred_timesteps > 1:
            dt = 0.1
            pos_diff = np.diff(valid_pred[:, :, :2], axis=1)
            vel = pos_diff / dt
            vel_mask = np.abs(vel) < 100
            filtered_vel = np.where(vel_mask, vel, 0)
            future[:n_pred_agents, 2 : min(n_pred_timesteps + 1, vel.shape[1] + 2), 3:5] = (
                filtered_vel[:, : min(n_pred_timesteps - 1, vel.shape[1])]
            )
        return future

    pred_futures = [(title, make_future_tensor(traj)) for title, traj in pred_trajs]

    images: List[np.ndarray] = []
    total_timesteps = agents_future.shape[1]

    # Use the FIRST prediction (Baseline or ControlNet) to track ego, not GT
    # This ensures camera follows actual predicted trajectory in closed-loop
    if pred_trajs and len(pred_trajs) > 0:
        # Use first prediction (usually Baseline)
        ego_pos = pred_trajs[0][1][0, :, :2]  # [T, 2] - ego agent's (x, y)
    else:
        # Fallback to GT if no predictions
        ego_pos = agents_future[0, :, :2]

    ego_valid = (
        (ego_pos[:, 0] != 0)
        & (ego_pos[:, 0] != -1)
        & (np.abs(ego_pos[:, 0]) < COORD_LIMIT)
        & (np.abs(ego_pos[:, 1]) < COORD_LIMIT)
    )

    center_x = 0.0
    center_y = 0.0
    if np.any(ego_valid):
        first_valid = np.where(ego_valid)[0][0]
        center_x, center_y = ego_pos[first_valid]

    def plot_frame(
        ax: plt.Axes,
        future_full: np.ndarray,
        future_masked: np.ndarray,
        title: str,
        future_color: str,
        timestep: int,
        conditioning_future: np.ndarray | None = None,
    ) -> None:
        ax.set_facecolor('#111111')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(center_x - VIEW_RANGE, center_x + VIEW_RANGE)
        ax.set_ylim(center_y - VIEW_RANGE, center_y + VIEW_RANGE)
        ax.set_title(title, fontsize=12, pad=12, color='#ECEFF1')
        ax.set_xlabel('X (m)', color='#B0BEC5')
        ax.set_ylabel('Y (m)', color='#B0BEC5')
        ax.tick_params(colors='#90A4AE', labelsize=8)
        ax.grid(color='#263238', linestyle='--', linewidth=0.4, alpha=0.6)

        for poly in polylines:
            if poly[0, 0] != 0:
                valid = (
                    (poly[:, 0] != 0)
                    & (np.abs(poly[:, 0]) < COORD_LIMIT)
                    & (np.abs(poly[:, 1]) < COORD_LIMIT)
                )
                if np.any(valid):
                    ax.plot(
                        poly[valid][:, 0],
                        poly[valid][:, 1],
                        color=palette['map'],
                        linewidth=0.6,
                        alpha=0.45,
                    )

        num_agents = min(agents_history.shape[0], future_full.shape[0])

        for i in range(num_agents):
            if timestep < future_full.shape[1]:
                curr = future_full[i, timestep]
                if (
                    curr[0] != 0
                    and curr[0] != -1
                    and np.abs(curr[0]) < COORD_LIMIT
                    and np.abs(curr[1]) < COORD_LIMIT
                ):
                    heading = curr[2] if len(curr) > 2 and curr[2] != 0 else agents_history[i, -1, 2]

                    # Determine box color based on agent type
                    is_controlled = controlled_agent_ids is not None and i in controlled_agent_ids
                    if i == 0:
                        box_color = palette['ego_box']
                        box_alpha = 0.9
                    elif is_controlled:
                        box_color = palette['controlled_box']
                        box_alpha = 0.85
                    else:
                        box_color = palette['others_box']
                        box_alpha = 0.5

                    rect = plt.Rectangle(
                        (curr[0] - agents_history[i, -1, 5] / 2, curr[1] - agents_history[i, -1, 6] / 2),
                        agents_history[i, -1, 5],
                        agents_history[i, -1, 6],
                        linewidth=2,
                        color=box_color,
                        alpha=box_alpha,
                        zorder=5,
                        transform=transforms.Affine2D().rotate_around(curr[0], curr[1], heading) + ax.transData,
                    )
                    ax.add_patch(rect)

                    # Add agent ID label
                    ax.text(
                        curr[0], curr[1],
                        str(i),
                        color='white',
                        fontsize=5,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none')
                    )

            future = future_masked[i, timestep:]
            valid_future = (
                (future[:, 0] != 0)
                & (future[:, 0] != -1)
                & (np.abs(future[:, 0]) < COORD_LIMIT)
                & (np.abs(future[:, 1]) < COORD_LIMIT)
            )
            if np.sum(valid_future) > 1:
                seg = future[valid_future]
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=future_color,
                    linewidth=2.2,
                    alpha=0.95,
                    linestyle='--',
                )

            if (
                conditioning_future is not None
                and i < conditioning_future.shape[0]
                and timestep < conditioning_future.shape[1]
            ):
                cond_future = conditioning_future[i, timestep:]
                valid_cond = (
                    (cond_future[:, 0] != 0)
                    & (cond_future[:, 0] != -1)
                    & (np.abs(cond_future[:, 0]) < COORD_LIMIT)
                    & (np.abs(cond_future[:, 1]) < COORD_LIMIT)
                )
                if np.sum(valid_cond) > 1:
                    cond_seg = cond_future[valid_cond]
                    ax.plot(
                        cond_seg[:, 0],
                        cond_seg[:, 1],
                        color=palette['condition_future'],
                        linewidth=2.0,
                        alpha=0.7,
                        linestyle='-',
                    )

        # IMPROVED: Show traffic lights with state-based colors
        # State mapping: 0=unknown, 1/4/7=red, 2/5/8=yellow, 3/6=green
        if traffic_lights.shape[1] >= 3:
            # Extract colors based on state
            colors = []
            for tl in traffic_lights:
                state = int(tl[2])
                if state in [1, 4, 7]:  # STOP states
                    colors.append('#FF5252')  # Red
                elif state in [2, 5, 8]:  # CAUTION states
                    colors.append('#FFC107')  # Yellow/Amber
                elif state in [3, 6]:  # GO states
                    colors.append('#4CAF50')  # Green
                else:  # UNKNOWN
                    colors.append('#9E9E9E')  # Gray
            ax.scatter(
                traffic_lights[:, 0],
                traffic_lights[:, 1],
                c=colors,
                s=20,
                alpha=0.85,
                edgecolors='none',
                zorder=4,
            )
        else:
            # Fallback: all red if no state info
            ax.scatter(
                traffic_lights[:, 0],
                traffic_lights[:, 1],
                c='#FF5252',
                s=20,
                alpha=0.85,
                edgecolors='none',
                zorder=4,
            )

    panel_titles: List[Tuple[str, np.ndarray, str]] = [
        (f"{scenario_name} — Ground Truth", agents_future, palette['gt_future'])
    ]
    alt_colors = [
        palette['pred_future'],
        palette['pred_future_alt'],
        palette['condition_future'],
        '#4DD0E1',
    ]
    for idx, (title, future) in enumerate(pred_futures):
        panel_titles.append((title, future, alt_colors[idx % len(alt_colors)]))

    # Pre-calculate layout parameters on first frame to ensure consistent size
    layout_params = None
    fixed_dpi = 100  # Fixed DPI for consistent pixel dimensions

    for t in range(0, total_timesteps, 2):
        if np.any(ego_valid):
            idx = min(t, len(ego_pos) - 1)
            while idx >= 0 and not ego_valid[idx]:
                idx -= 1
            if idx >= 0:
                center_x, center_y = ego_pos[idx]

        fig, axes = plt.subplots(1, len(panel_titles), figsize=(6 * len(panel_titles), 6), dpi=fixed_dpi)
        axes = np.atleast_1d(axes)

        gt_partial = agents_future.copy()
        if t + 1 < gt_partial.shape[1]:
            gt_partial[:, t + 1 :] = -1

        for ax, (title, future_full, color) in zip(axes, panel_titles):
            partial = future_full.copy()
            if t + 1 < partial.shape[1]:
                partial[:, t + 1 :] = -1
            conditioning_future = None
            if conditioning_futures is not None:
                conditioning_future = conditioning_futures.get(title)
            plot_frame(
                ax,
                future_full,
                partial,
                f"{title} · t={t * 0.1:.1f}s",
                color,
                t,
                conditioning_future=conditioning_future,
            )

        # Use cached layout params after first frame
        if layout_params is None:
            fig.tight_layout(pad=1.2)
            # Cache the layout parameters from first frame
            layout_params = {
                'left': fig.subplotpars.left,
                'right': fig.subplotpars.right,
                'top': fig.subplotpars.top,
                'bottom': fig.subplotpars.bottom,
                'wspace': fig.subplotpars.wspace,
                'hspace': fig.subplotpars.hspace,
            }
        else:
            # Reuse cached layout for consistent frame size
            fig.subplots_adjust(**layout_params)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img[:, :, :3].copy())
        plt.close(fig)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save as GIF
        gif_path = f"{save_path}.gif"
        pil_images = [Image.fromarray(img) for img in images]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=100,  # 100ms per frame = 10 fps
            loop=0,  # Loop forever
            optimize=False,  # Set to True for smaller file size but slower
        )
        print(f"Saved GIF to {gif_path}")

    plt.rcParams['axes.prop_cycle'] = original_style
    plt.style.use('default')
    return images

# -----------------------------------------------------------------------------
# Map Bounds and Validity Checking Functions
# -----------------------------------------------------------------------------
def compute_map_bounds_from_polylines(
    batch: dict,
    margin_meters: float = 10.0,
    fallback_bounds: dict = None,
) -> dict:
    """
    從 road network (polylines) 計算地圖的 axis-aligned bounding box。

    這個方法找出所有 road points 的最小外接矩形，代表實際可行駛區域。

    Args:
        batch: Model batch containing 'polylines' [B, N_poly, N_points, D]
        margin_meters: 安全邊界（米），加在 bounding box 外圍
        fallback_bounds: 如果無法計算時使用的預設邊界

    Returns:
        dict: {'x_min', 'x_max', 'y_min', 'y_max'} 地圖邊界（單位：米）
    """
    if fallback_bounds is None:
        fallback_bounds = {
            'x_min': -150.0, 'x_max': 150.0,
            'y_min': -150.0, 'y_max': 150.0,
        }

    if 'polylines' not in batch:
        return fallback_bounds

    polylines = batch['polylines']  # [B, N_poly, N_points, D]

    # Convert to numpy if needed
    if isinstance(polylines, torch.Tensor):
        polylines = polylines.cpu().numpy()

    # Extract first batch (assuming batch_size=1)
    if polylines.ndim == 4:
        polylines = polylines[0]  # [N_poly, N_points, D]

    # 提取有效的 polyline 點（非零點）
    # polylines 格式: [N_poly, N_points, D] where D >= 2 (x, y, ...)
    valid_mask = polylines.sum(-1) != 0  # [N_poly, N_points]

    if not valid_mask.any():
        return fallback_bounds

    # 提取所有有效點的 x, y 座標
    valid_points = polylines[valid_mask]  # [N_valid, D]
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]

    # 計算 bounding box
    x_min = float(x_coords.min())
    x_max = float(x_coords.max())
    y_min = float(y_coords.min())
    y_max = float(y_coords.max())

    # 加上安全邊界
    bounds = {
        'x_min': x_min - margin_meters,
        'x_max': x_max + margin_meters,
        'y_min': y_min - margin_meters,
        'y_max': y_max + margin_meters,
    }

    return bounds


def check_agents_within_bounds(
    pred_trajs: torch.Tensor | np.ndarray,
    map_bounds: dict,
) -> torch.Tensor | np.ndarray:
    """
    Check if predicted trajectories are within map bounds (Supports Tensor and NumPy).
    """
    if isinstance(pred_trajs, torch.Tensor):
        x = pred_trajs[..., 0]
        y = pred_trajs[..., 1]
        
        in_bounds = (
            (x >= map_bounds['x_min']) & (x <= map_bounds['x_max']) &
            (y >= map_bounds['y_min']) & (y <= map_bounds['y_max'])
        )
        # Cumulative invalid using cumprod (1=valid, 0=invalid)
        valid_mask = torch.cumprod(in_bounds.long(), dim=-1).bool()
        return valid_mask
    else:
        # Fallback to NumPy implementation
        # Handle both [B, A, T, D] and [A, T, D]
        if pred_trajs.ndim == 3:
            pred_trajs = pred_trajs[None, ...]  # [1, A, T, D]
            squeeze_output = True
        else:
            squeeze_output = False

        x = pred_trajs[..., 0]  # [B, A, T]
        y = pred_trajs[..., 1]  # [B, A, T]

        # Check bounds
        in_bounds = (
            (x >= map_bounds['x_min']) & (x <= map_bounds['x_max']) &
            (y >= map_bounds['y_min']) & (y <= map_bounds['y_max'])
        )  # [B, A, T]

        # Once invalid, always invalid logic
        # np.maximum.accumulate on inverse mask propagates True (invalid) forward
        # valid = NOT (accumulated invalid)
        invalid_accumulated = np.maximum.accumulate(~in_bounds, axis=-1)
        valid_mask = ~invalid_accumulated

        if squeeze_output:
            return valid_mask[0]
        return valid_mask


# -----------------------------------------------------------------------------
# Closed-Loop Environment Update Functions
# -----------------------------------------------------------------------------
def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi] (PyTorch/NumPy compatible).
    """
    if isinstance(angle, torch.Tensor):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_relations_batch(agents, polylines, traffic_lights):
    """
    Calculate relations for a batch of scenarios (Vectorized, PyTorch).
    
    Args:
        agents: [B, A, H, D]
        polylines: [B, N_poly, N_pts, D]
        traffic_lights: [B, N_tl, D]
        
    Returns:
        relations: [B, N_total, N_total, 3]
    """
    # Ensure inputs are tensors
    if isinstance(agents, np.ndarray):
        # Fallback for legacy numpy calls if any
        return calculate_relations_batch_numpy(agents, polylines, traffic_lights)

    B, n_agents = agents.shape[:2]
    n_polylines = polylines.shape[1]
    n_traffic_lights = traffic_lights.shape[1]
    n = n_agents + n_polylines + n_traffic_lights
    device = agents.device
    
    # Extract relevant parts
    agents_curr = agents[:, :, -1, :3] # [B, A, 3]
    polys = polylines[:, :, 0, :3]     # [B, N_poly, 3]

    if traffic_lights.shape[2] >= 3:
        tls_3d = traffic_lights[:, :, :3]
    else:
        tls = traffic_lights[:, :, :2]
        tls_3d = torch.cat([tls, torch.zeros((B, n_traffic_lights, 1), device=device, dtype=tls.dtype)], dim=2)

    all_elements = torch.cat([agents_curr, polys, tls_3d], dim=1) # [B, N, 3]
    
    # Pairwise diff
    # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
    pos_diff = all_elements[:, :, None, :2] - all_elements[:, None, :, :2]
    
    headings = all_elements[:, :, 2:3]
    cos_theta = torch.cos(headings)
    sin_theta = torch.sin(headings)
    
    dx = pos_diff[..., 0]
    dy = pos_diff[..., 1]
    
    local_pos_x = dx * cos_theta + dy * sin_theta
    local_pos_y = -dx * sin_theta + dy * cos_theta
    
    theta_i = headings
    theta_j = headings.transpose(1, 2)
    theta_diff = wrap_to_pi(theta_i - theta_j)
    
    # Traffic light mask
    start_idx = n_agents + n_polylines
    indices = torch.arange(n, device=device)
    is_tl = indices >= start_idx
    tl_mask = is_tl[:, None] | is_tl[None, :] # [N, N]
    
    theta_diff = torch.where(tl_mask[None, ...], torch.zeros_like(theta_diff), theta_diff)
    
    # Diagonal epsilon
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)[None, ...]
    epsilon = 0.01
    
    # Use tensor scalar for where
    eps_tensor = torch.tensor(epsilon, device=device, dtype=local_pos_x.dtype)
    local_pos_x = torch.where(diag_mask, eps_tensor, local_pos_x)
    local_pos_y = torch.where(diag_mask, eps_tensor, local_pos_y)
    theta_diff = torch.where(diag_mask, eps_tensor, theta_diff)
    
    # Zero mask (x==0)
    x_coords = all_elements[..., 0]
    zero_mask = (x_coords[:, :, None] == 0) | (x_coords[:, None, :] == 0)
    
    relations = torch.stack([local_pos_x, local_pos_y, theta_diff], dim=-1)
    relations = torch.where(zero_mask[..., None], torch.zeros_like(relations), relations)
    
    return relations.float()


def calculate_relations_batch_numpy(agents, polylines, traffic_lights):
    """Legacy NumPy implementation for compatibility."""
    B = agents.shape[0]
    n_agents = agents.shape[1]
    n_polylines = polylines.shape[1]
    n_traffic_lights = traffic_lights.shape[1]
    n = n_agents + n_polylines + n_traffic_lights
    
    # Extract relevant parts
    # Agents: use last history step [B, A, 3] (x,y,yaw)
    agents_curr = agents[:, :, -1, :3]
    
    # Polylines: use first point [B, N_poly, 3]
    polys = polylines[:, :, 0, :3]

    # Traffic lights: [B, N_tl, 3] - FIXED: use state (column 2) instead of discarding it
    if traffic_lights.shape[2] >= 3:
        tls_3d = traffic_lights[:, :, :3]  # [x, y, state]
    else:
        # Fallback for old data without state
        tls = traffic_lights[:, :, :2]
        tls_3d = np.concatenate([tls, np.zeros((B, n_traffic_lights, 1), dtype=tls.dtype)], axis=2)

    # Concatenate all elements: [B, N, 3]
    all_elements = np.concatenate([agents_curr, polys, tls_3d], axis=1)
    
    # Compute pairwise differences using broadcasting
    # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
    pos_diff = all_elements[:, :, None, :2] - all_elements[:, None, :, :2]
    
    # Compute local positions and angle differences
    # headings: [B, N, 1]
    headings = all_elements[:, :, 2:3]
    cos_theta = np.cos(headings)
    sin_theta = np.sin(headings)
    
    # Broadcast for rotation: [B, N, N]
    # We project diff(j-i) onto i's frame.
    # pos_diff[b, i, j] is (xj - xi, yj - yi)
    # We want to rotate this vector by -theta_i
    # local_x = dx * cos(th) + dy * sin(th)
    # local_y = -dx * sin(th) + dy * cos(th)
    
    # align dimensions:
    # cos_theta: [B, N, 1] -> [B, N, 1]
    # pos_diff: [B, N, N, 2]
    
    dx = pos_diff[..., 0] # [B, N, N]
    dy = pos_diff[..., 1] # [B, N, N]
    
    local_pos_x = dx * cos_theta + dy * sin_theta
    local_pos_y = -dx * sin_theta + dy * cos_theta
    
    # Angle diff: theta_i - theta_j (matches original implementation)
    # [B, N, 1] - [B, 1, N] -> [B, N, N]
    theta_i = headings # [B, N, 1]
    theta_j = headings.transpose(0, 2, 1) # [B, 1, N]
    theta_diff = wrap_to_pi(theta_i - theta_j)
    
    # Set theta_diff to zero for traffic lights (as in original)
    start_idx = n_agents + n_polylines
    # Mask for TL: indices >= start_idx
    # In batch, range is same for all b
    indices = np.arange(n)
    is_tl = indices >= start_idx
    # if i is TL or j is TL, zero out theta_diff?
    # Original: np.where((indices >= start_idx)[:, None] | (indices >= start_idx)[None, :], 0, theta_diff)
    # Broadcasting [N, 1] | [1, N] -> [N, N]
    tl_mask = is_tl[:, None] | is_tl[None, :] # [N, N]
    # Broadcast to batch [B, N, N]
    theta_diff = np.where(tl_mask[None, ...], 0.0, theta_diff)
    
    # Set diagonal to epsilon
    diag_mask = np.eye(n, dtype=bool)[None, ...] # [1, N, N]
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)
    
    # Zero mask (if element 0 is 0... actually checking x==0)
    # Original: all_elements[:, 0][:, None] == 0 ...
    # all_elements is [B, N, 3].
    # Check if x is 0.
    x_coords = all_elements[..., 0] # [B, N]
    zero_mask = (x_coords[:, :, None] == 0) | (x_coords[:, None, :] == 0) # [B, N, N]
    
    # Stack relations: [B, N, N, 3]
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)
    
    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)
    
    return relations.astype(np.float32)


def update_batch_with_prediction(
    batch: dict[str, torch.Tensor],
    pred_trajs: torch.Tensor | np.ndarray,
    horizon_steps: int = 11,
    dt: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Update batch history with predicted trajectories (PyTorch version).

    Batched version of the original single-scenario update function.
    Handles [B, A, T, C] predictions for B parallel rollouts.

    Args:
        batch: Current batch with 'agents_history' [B, A, H, D]
        pred_trajs: Predicted trajectories [B, A, T, C] where C=3 (x,y,heading) or C=5 (x,y,heading,vx,vy)
        horizon_steps: Number of steps to execute
        dt: Time step

    Returns:
        Updated batch
    """
    agents_history = batch['agents_history']  # [B, A, H, D]
    batch_size, num_agents, history_len, state_dim = agents_history.shape
    device = agents_history.device

    # Ensure pred_trajs is a tensor on the correct device
    if isinstance(pred_trajs, np.ndarray):
        pred_trajs = torch.from_numpy(pred_trajs).to(device)
    elif pred_trajs.device != device:
        pred_trajs = pred_trajs.to(device)

    # 1. Extract executed part from predictions: [B, A, T, C] -> [B, A, horizon, C]
    executed_trajs = pred_trajs[:, :, :horizon_steps, :]

    # 2. Create new history states for all batch items
    new_history_states = torch.zeros((batch_size, num_agents, horizon_steps, state_dim), dtype=agents_history.dtype, device=device)

    # Copy position and heading (dimensions 0, 1, 2)
    new_history_states[..., 0] = executed_trajs[..., 0]  # x
    new_history_states[..., 1] = executed_trajs[..., 1]  # y
    
    if executed_trajs.shape[3] > 2:
        new_history_states[..., 2] = executed_trajs[..., 2]  # heading

    # Velocity: check if model outputs it directly
    if executed_trajs.shape[3] >= 5:
        # Model outputs velocity: use it directly
        new_history_states[..., 3] = executed_trajs[..., 3]  # vx
        new_history_states[..., 4] = executed_trajs[..., 4]  # vy
    else:
        # Fallback: Calculate velocity from position differences
        # Use the last history point to compute velocity for the first new point
        last_pos = agents_history[:, :, -1, :2]  # [B, A, 2]
        first_new_pos = executed_trajs[:, :, 0, :2]  # [B, A, 2]

        # Velocity at t=0 (first new step) = (p_0 - p_history_last) / dt
        v0 = (first_new_pos - last_pos) / dt
        new_history_states[:, :, 0, 3] = v0[..., 0]
        new_history_states[:, :, 0, 4] = v0[..., 1]

        if horizon_steps > 1:
            # Calculate velocity from position diff within prediction for t=1...
            dx = torch.diff(executed_trajs[..., 0], dim=2) / dt  # [B, A, horizon-1]
            dy = torch.diff(executed_trajs[..., 1], dim=2) / dt  # [B, A, horizon-1]
            # Assign to frames 1 onwards
            new_history_states[:, :, 1:, 3] = dx
            new_history_states[:, :, 1:, 4] = dy

    # Copy vehicle dimensions (length, width) from last history timestep
    # agents_history: [B, A, H, D], take last timestep [-1]
    # Broadcast [B, A, 2] -> [B, A, 1, 2] -> [B, A, horizon, 2]
    # Note: unsqueeze(2) adds the time dimension for broadcasting
    new_history_states[..., 5:7] = agents_history[:, :, -1, 5:7].unsqueeze(2)

    # Copy other dimensions if present (e.g., height, type, etc.)
    if state_dim > 7:
        new_history_states[..., 7:] = agents_history[:, :, -1, 7:].unsqueeze(2)

    # 3. Shift history: drop oldest horizon_steps, append new predictions
    # [B, A, H-horizon, D] concat [B, A, horizon, D] -> [B, A, H, D]
    # agents_history[:, :, horizon_steps:, :] drops the first 'horizon_steps' (oldest)
    updated_history = torch.cat([
        agents_history[:, :, horizon_steps:, :],
        new_history_states
    ], dim=2)

    # Ensure correct length (safety check)
    if updated_history.shape[2] > history_len:
        updated_history = updated_history[:, :, -history_len:, :]
    elif updated_history.shape[2] < history_len:
        # Pad if needed
        pad_len = history_len - updated_history.shape[2]
        pad = torch.zeros((batch_size, num_agents, pad_len, state_dim), dtype=agents_history.dtype, device=device)
        updated_history = torch.cat([pad, updated_history], dim=2)

    # 4. Update batch with new history
    batch['agents_history'] = updated_history

    # 5. Recalculate relations for each batch item (VECTORIZED)
    # Note: calculate_relations_batch now supports tensors
    
    # Ensure polylines and traffic lights are tensors
    if isinstance(batch['polylines'], np.ndarray):
        batch['polylines'] = torch.from_numpy(batch['polylines']).to(device)
    if isinstance(batch['traffic_light_points'], np.ndarray):
        batch['traffic_light_points'] = torch.from_numpy(batch['traffic_light_points']).to(device)

    new_relations = calculate_relations_batch(
        updated_history, 
        batch['polylines'], 
        batch['traffic_light_points']
    )
    
    batch['relations'] = new_relations

    return batch


def update_sketch_conditioning(
    batch: dict[str, torch.Tensor],
    original_sketch: np.ndarray | None,
    executed_steps: int,
) -> dict[str, torch.Tensor]:
    """
    Update sketch conditioning for batch (Vectorized).
    """
    if 'agents_future' not in batch or original_sketch is None:
        return batch

    agents_future = batch['agents_future']  # [B, A, T, C]
    batch_size, num_agents, future_len, future_dim = agents_future.shape

    original_len = original_sketch.shape[1]

    if executed_steps >= original_len:
        agents_future[..., :original_sketch.shape[2]] = 0.0
    else:
        remaining_sketch = original_sketch[:, executed_steps:, :].copy()
        remaining_len = remaining_sketch.shape[1]

        if remaining_len < future_len:
            pad_len = future_len - remaining_len
            pad = np.zeros((num_agents, pad_len, remaining_sketch.shape[2]), dtype=np.float32)
            padded_sketch = np.concatenate([remaining_sketch, pad], axis=1)
        else:
            padded_sketch = remaining_sketch[:, :future_len, :]

        # Update all batch elements (assuming same sketch for all rollouts of same scenario)
        sketch_tensor = torch.from_numpy(padded_sketch).to(agents_future.device)
        agents_future[..., :original_sketch.shape[2]] = sketch_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)

    batch['agents_future'] = agents_future
    return batch


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Batch Rollout Runner
# -----------------------------------------------------------------------------
def run_batch_rollouts(
    model,
    model_name: str,
    initial_batch: dict,
    args: argparse.Namespace,
    horizon_steps: int,
    total_steps: int,
    num_iterations: int,
    map_bounds: dict,
    start_seed: int,
    num_rollouts: int,
    gt_future: np.ndarray | None = None,
    conditioning_latents: dict | None = None,
    original_sketch: np.ndarray | None = None,
    is_latent_mode: bool = False,
    num_candidates: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run multiple closed-loop rollouts in PARALLEL as a batch.

    Args:
        num_rollouts: Number of parallel rollouts (batch size).
        ...

    Returns:
        trajs: [num_rollouts, A, T, 4]
        valid_masks: [num_rollouts, A, T]
    """
    # 1. Replicate batch for parallel rollouts
    # initial_batch keys are [1, ...]. We want [num_rollouts, ...]
    model_batch = {}
    for key, value in initial_batch.items():
        if isinstance(value, torch.Tensor):
            # Expand [1, ...] to [num_rollouts, ...]
            # Use expand if possible to save memory, but cat is safer for modification
            model_batch[key] = value.repeat(num_rollouts, *([1] * (value.ndim - 1))).to(args.device)
        else:
            model_batch[key] = value # Non-tensor copy reference


    batch_size = num_rollouts
    num_agents = model_batch['agents_history'].shape[1]
    device = model_batch['agents_history'].device

    # Extract z-coordinates
    if gt_future is not None and gt_future.shape[-1] > 7:
        gt_z = gt_future[:, :total_steps, 7]  # [A, T]
    else:
        agents_history_np = initial_batch['agents_history'][0].cpu().numpy()
        if agents_history_np.shape[-1] > 7:
            initial_z = agents_history_np[:, -1, 7]
            gt_z = np.tile(initial_z[:, np.newaxis], (1, total_steps))
        else:
            gt_z = np.zeros((num_agents, total_steps), dtype=np.float32)
    
    # Expand gt_z for batch: [B, A, T]
    gt_z_batch_np = np.tile(gt_z[None, ...], (batch_size, 1, 1))
    gt_z_batch = torch.from_numpy(gt_z_batch_np).to(device)

    # Storage
    all_executed_trajs = [] # List of [B, A, steps, 4]
    agents_valid = torch.ones((batch_size, num_agents), dtype=torch.bool, device=device)

    # Closed-loop iterations
    for iter_idx in range(num_iterations):
        current_step = iter_idx * horizon_steps
        remaining_steps = min(horizon_steps, total_steps - current_step)

        if remaining_steps <= 0:
            break

        # Predict future
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):     
            if args.test_mode == 'diffusion':
                # Run diffusion with num_samples=8 candidates per rollout
                # Input batch: [B, ...], Output candidates: [B*8, ...] (if duplicate_batch used cat)
                # We need to check duplicate_batch implementation in utils.py.
                # Assuming duplicate_batch repeats: [b1, b2] -> [b1, b2, b1, b2] or [b1, b1, b2, b2]
                # Based on standard impl, usually [b1, b2]*N -> [b1, b2, b1, b2...]
                # So output denoised_trajs is [B*8, A, T, D]
                
                # Support DataParallel wrapper
                if isinstance(model, torch.nn.DataParallel):
                    pred = model(
                        model_batch,
                        mode='sample_denoiser',
                        num_samples=num_candidates,
                        use_tqdm=False,
                        replan_idx=iter_idx,
                    )
                else:
                    pred = model.sample_denoiser(
                        model_batch,
                        num_samples=num_candidates,
                        use_tqdm=False,
                        replan_idx=iter_idx,
                    )
                
                all_trajs = pred['denoised_trajs'] # [B*8, A, T, D]

                # We need to reshape to [8, B, A, T, D] for select_safest_trajectory
                # duplicate_batch uses repeat_interleave: [b1_c1, b1_c2... b2_c1, b2_c2...]
                # So the first dimension is (batch_size * num_candidates)
                
                # 1. Recover structure [B, M, A, T, D]
                all_trajs = all_trajs.view(batch_size, num_candidates, num_agents, -1, all_trajs.shape[-1])
                
                # 2. Transpose to [M, B, A, T, D]
                all_trajs = all_trajs.transpose(0, 1)

                # Prepare agent dims [B, A, 2]
                agent_dims = model_batch['agents_history'][:, :, -1, 5:7]
                
                # Valid mask [B, A]
                agent_valid_mask = agents_valid.float()

                # Select safest [B, A, T, D]
                best_traj_tensor = select_safest_trajectory(all_trajs, agent_dims, agent_valid_mask=agent_valid_mask)
                pred_traj = best_traj_tensor # Keep on GPU as tensor

            elif args.test_mode == 'prior':
                 # Prior mode logic (simplified, usually deterministic or simple sampling)
                 if isinstance(model, torch.nn.DataParallel):
                     pred = model(model_batch, mode='inference_predictor')
                 else:
                     pred = model.inference_predictor(model_batch)
                 
                 # pred['goal_trajs']: [B, A, K, T, D]
                 # Just pick one or top 1.
                 # ... (Prior mode implementation simplified for brevity, assuming diffusion focus)
                 trajs = pred['goal_trajs'][:, :, 0] # [B, A, T, D] top 1
                 pred_traj = trajs # Keep on GPU
            else:
                raise NotImplementedError

        # Check validity [B, A, remaining_steps]
        validity_mask = check_agents_within_bounds(
            pred_traj[:, :, :remaining_steps, :],
            map_bounds
        )

        # Update agents_valid (cumulative)
        # If any step invalid in this segment, mark agent invalid? 
        # check_agents_within_bounds returns False for all steps AFTER first violation.
        # So if last step is invalid, agent is invalid? 
        # We just need to detect if ANY False occurred.
        segment_valid = validity_mask.all(dim=-1) # [B, A]
        agents_valid = agents_valid & segment_valid

        # Format segment [B, A, steps, 4]
        executed_segment = torch.zeros((batch_size, num_agents, remaining_steps, 4), dtype=pred_traj.dtype, device=device)
        executed_segment[..., 0] = pred_traj[..., :remaining_steps, 0]
        executed_segment[..., 1] = pred_traj[..., :remaining_steps, 1]
        if pred_traj.shape[-1] > 2:
            executed_segment[..., 3] = pred_traj[..., :remaining_steps, 2]
        
        # Z coords
        z_start = current_step
        z_end = current_step + remaining_steps
        executed_segment[..., 2] = gt_z_batch[..., z_start:z_end]

        # Handle invalid steps (freeze)
        # For batch, looping is slow. Vectorize?
        # validity_mask: [B, A, T]. True=Valid.
        # If invalid, replace with last valid?
        # Complex to vectorize efficiently in numpy without creating copies.
        # Let's use a loop over B/A since numpy is fast enough for 32x32.
        # for b in range(batch_size):
        #     for a in range(num_agents):
        #         if not validity_mask[b, a].all():
        #             # invalid_indices = np.where(~validity_mask[b, a])[0]
        #             # first_invalid = invalid_indices[0]
        #             # But check_agents_within_bounds already ensures contiguous invalidity.
        #             # So just find first False.
        #             first_invalid = np.argmax(~validity_mask[b, a])
        #             if ~validity_mask[b, a][first_invalid]: # Check if actually found
        #                 if first_invalid > 0:
        #                      last_valid_pos = executed_segment[b, a, first_invalid-1]
        #                      executed_segment[b, a, first_invalid:] = last_valid_pos
        #                 # else: all invalid, keep prediction (or zero?)

        all_executed_trajs.append(executed_segment)

        # Update environment
        if iter_idx < num_iterations - 1:
             saved_agent_latents = model_batch.get('agent_latents', None)
             saved_scenario_latents = model_batch.get('scenario_latents', None)

             model_batch = update_batch_with_prediction(
                 batch=model_batch,
                 pred_trajs=pred_traj,
                 horizon_steps=remaining_steps
             )

             # Restore conditioning
             if model_name == 'controlnet':
                if is_latent_mode:
                     # Logic for latent segments...
                     if 'agent_latents_segments' in model_batch:
                         # Need to handle batch selection if segments are shared
                         # Here segments are shared across rollouts, so just pick correct segment
                         # and repeat for batch. If we run past the last segment, drop conditioning.
                         num_segments = model_batch.get('num_segments', None)
                         if num_segments is None:
                             num_segments = model_batch['agent_latents_segments'].shape[0]
                         elif isinstance(num_segments, torch.Tensor):
                             num_segments = int(num_segments.item())
                         else:
                             num_segments = int(num_segments)

                         latent_index_mode = model_batch.get('latent_index_mode', 'time')
                         if isinstance(latent_index_mode, torch.Tensor):
                             latent_index_mode = latent_index_mode.item()

                         if latent_index_mode == 'window':
                             seg_idx = iter_idx + 1
                         else:
                             latent_seg_len = int(model_batch.get('latent_segment_len', horizon_steps))
                             current_time_step = (iter_idx + 1) * horizon_steps
                             seg_idx = current_time_step // latent_seg_len

                         if seg_idx >= num_segments:
                             # No conditioning past the last segment.
                             ag_lat = torch.zeros_like(model_batch['agent_latents_segments'][0])
                             sc_lat = None
                             if 'scenario_latents_segments' in model_batch:
                                 sc_lat = torch.zeros_like(model_batch['scenario_latents_segments'][0])
                         else:
                             # Segments: [seg_len, 1, A, D] -> [seg_len, B, A, D]
                             # Actually stored as [Segs, 1, A, D].
                             ag_lat = model_batch['agent_latents_segments'][seg_idx] # [1, A, D]
                             sc_lat = None
                             if 'scenario_latents_segments' in model_batch:
                                 sc_lat = model_batch['scenario_latents_segments'][seg_idx]

                         model_batch['agent_latents'] = ag_lat.repeat(batch_size, 1, 1)
                         if sc_lat is not None:
                             model_batch['scenario_latents'] = sc_lat.repeat(batch_size, 1, 1)
                         elif 'scenario_latents' in model_batch:
                             model_batch['scenario_latents'] = torch.zeros_like(model_batch['scenario_latents'])
                     else:
                         if saved_agent_latents is not None:
                             model_batch['agent_latents'] = saved_agent_latents
                         if saved_scenario_latents is not None:
                             model_batch['scenario_latents'] = saved_scenario_latents
                elif original_sketch is not None:
                    total_executed = (iter_idx + 1) * horizon_steps
                    model_batch = update_sketch_conditioning(
                        model_batch, original_sketch, total_executed
                    )
    
    # Concatenate [B, A, T_total, 4]
    full_traj = torch.cat(all_executed_trajs, dim=2)

    # Create final valid mask
    # Recompute based on full traj zeros check?
    # Or use the accumulated valid mask logic?
    # Let's use the zero check to match original logic exactly
    # full_valid_mask = np.ones((batch_size, num_agents, total_steps), dtype=bool)
    
    # Vectorized zero check
    # traj [B, A, T, 4]
    # Check if all 4 dims are 0 (exact 0.0)
    is_zero = torch.all(full_traj == 0.0, dim=-1) # [B, A, T]
    
    # If t is zero, then t+1... are invalid.
    # Using cumsum logic
    is_zero_cum = torch.cummax(is_zero.long(), dim=-1)[0].bool()
    full_valid_mask = ~is_zero_cum
    
    return full_traj.cpu().numpy(), full_valid_mask.cpu().numpy()


# -----------------------------------------------------------------------------
# Closed-Loop Simulation Runner
# -----------------------------------------------------------------------------
def run_closedloop_simulation(
    args: argparse.Namespace,
    sketch_fn: SketchFn,
    sketch_name: str = 'Sketch',
    num_samples: int = 32,
    batch_size: int = 32, # NEW: control batch size
    batch_preprocess_fn = None,
) -> None:
    """
    Run Closed-Loop Simulation.
    """
    if args.model_path is None:
        raise ValueError("--model_path must be provided")
    if args.test_path is None:
        raise ValueError("--test_path must be provided")
    if not args.use_training_format:
        raise NotImplementedError("Only PKL format supported.")

    # Determine batch size: prioritize command-line argument over function parameter
    if hasattr(args, 'batch_size'):
        batch_size = args.batch_size
    
    # Determine num_samples: prioritize command-line argument
    if hasattr(args, 'num_samples'):
        num_samples = args.num_samples

    # Load models
    base_model = VBDTest.load_from_checkpoint(args.model_path, args.device)
    base_model.reset_agent_length(N_SIM_AGENTS)
    base_model.eval()

    control_model = None
    if args.ctrl_ckpt_path:
        control_model = VBDCtrlTest.load_from_checkpoint(
            args.ctrl_ckpt_path,
            map_location=args.device,
        )
        control_model.reset_agent_length(N_SIM_AGENTS)
        control_model.eval()

        # DataParallel disabled: batch size typically too small for efficient multi-GPU usage
        # For closed-loop with small batches, single GPU is more efficient
        # To enable: uncomment below and ensure batch_size >= 16
        # if torch.cuda.device_count() > 1 and args.device != 'cpu':
        #     print(f"🚀 Enabling DataParallel for ControlNet on {torch.cuda.device_count()} GPUs")
        #     control_model = torch.nn.DataParallel(control_model)
        #     control_model.to(args.device) # Ensure model is on the main device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    # Closed-loop settings
    horizon_steps = args.horizon_steps
    total_steps = args.total_steps
    num_iterations = (total_steps + horizon_steps - 1) // horizon_steps

    print(f"\n{'='*70}")
    print(f"CLOSED-LOOP SIMULATION")
    print(f"{'='*70}")
    print(f"Horizon: {horizon_steps} steps")
    print(f"Batch Size: {batch_size} (parallel rollouts)")
    print(f"Num Samples: {num_samples}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    pkl_files = load_pkl_files(args.test_path)
    if not pkl_files:
        raise RuntimeError(f"No PKL files found in {args.test_path}")

    if args.max_scenarios > 0:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(pkl_files))[:args.max_scenarios]
    else:
        order = np.arange(len(pkl_files))

    output_dir = getattr(args, 'output_dir', 'test_closedloop_1208')
    save_dir = os.path.join('testing_results', output_dir, str(args.seed))
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(order, desc="Processing scenarios"):
        scenario_path = pkl_files[idx]
        scenario_id = os.path.basename(scenario_path).replace('.pkl', '')

        try:
            scenario_data = load_pkl_data(scenario_path)
            batch = prepare_model_batch(scenario_data)
            if batch_preprocess_fn is not None:
                batch_preprocess_fn(batch, scenario_data)

            if getattr(args, "only_controlnet", False):
                if control_model is None:
                    raise ValueError("--only_controlnet requires --ctrl_ckpt_path")
                model_specs = [('controlnet', control_model, 'ControlNet (CL)')]
            else:
                model_specs = [('baseline', base_model, 'Baseline (CL)')]
                if control_model is not None:
                    model_specs.append(('controlnet', control_model, 'ControlNet (CL)'))

            all_model_trajectories = {}
            predictions_for_vis = []
            conditioning_futures = {}

            map_bounds = compute_map_bounds_from_polylines(batch, margin_meters=0.0)

            # Prepare sketch/latent
            batch_for_sketch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            conditioning_output = sketch_fn(batch_for_sketch)

            is_latent_mode = False
            original_sketch = None
            conditioning_latents = None
            controlled_agent_ids = []

            if isinstance(conditioning_output, tuple) and len(conditioning_output) == 2:
                first_element = conditioning_output[0]
                if isinstance(first_element, dict) and 'agent_latents' in first_element:
                    is_latent_mode = True
                    conditioning_latents, controlled_agent_ids = conditioning_output
                else:
                    original_sketch, controlled_agent_ids = conditioning_output
            else:
                if isinstance(conditioning_output, tuple):
                    original_sketch, controlled_agent_ids = conditioning_output
                else:
                    original_sketch = conditioning_output
                    controlled_agent_ids = []

            for model_idx, (model_name, model, model_title) in enumerate(model_specs):
                # Prepare Batch
                model_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Conditioning Setup
                if model_name == 'baseline':
                    if 'agents_future' in model_batch: model_batch['agents_future'].zero_()
                    if 'agent_latents' in model_batch: del model_batch['agent_latents']
                    if 'scenario_latents' in model_batch: del model_batch['scenario_latents']
                elif model_name == 'controlnet':
                    if is_latent_mode and conditioning_latents is not None:
                        if 'segment_latents' in conditioning_latents:
                            num_segments = conditioning_latents.get('num_segments', 3)
                            segment_latents_list = conditioning_latents['segment_latents']

                            # Stack segments [S, A, D] -> [S, 1, A, D]
                            agent_latents_segments = torch.stack(
                                [torch.from_numpy(s['agent_latents']).float() for s in segment_latents_list],
                                dim=0,
                            ).unsqueeze(1).to(args.device)
                            scenario_latents_segments = torch.stack(
                                [torch.from_numpy(s['scenario_latents']).float() for s in segment_latents_list],
                                dim=0,
                            ).unsqueeze(1).to(args.device)

                            model_batch['agent_latents_segments'] = agent_latents_segments
                            model_batch['scenario_latents_segments'] = scenario_latents_segments
                            model_batch['num_segments'] = num_segments
                            if 'latent_index_mode' in conditioning_latents:
                                model_batch['latent_index_mode'] = conditioning_latents['latent_index_mode']
                            if 'latent_segment_len' in conditioning_latents:
                                model_batch['latent_segment_len'] = conditioning_latents['latent_segment_len']
                            model_batch['agent_latents'] = agent_latents_segments[0]
                            model_batch['scenario_latents'] = scenario_latents_segments[0]
                        else:
                            model_batch['agent_latents'] = torch.from_numpy(
                                conditioning_latents['agent_latents']
                            ).unsqueeze(0).float().to(args.device)
                            model_batch['scenario_latents'] = torch.from_numpy(
                                conditioning_latents['scenario_latents']
                            ).unsqueeze(0).float().to(args.device)
                    elif original_sketch is not None and 'agents_future' in model_batch:
                        sketch_tensor = torch.from_numpy(original_sketch).float()
                        # Truncate/Pad
                        future_len = model_batch['agents_future'].shape[2]
                        if sketch_tensor.shape[1] > future_len:
                            sketch_tensor = sketch_tensor[:, :future_len, :]
                        elif sketch_tensor.shape[1] < future_len:
                            pad = torch.zeros(
                                model_batch['agents_future'].shape[1],
                                future_len - sketch_tensor.shape[1],
                                sketch_tensor.shape[2],
                            )
                            sketch_tensor = torch.cat([sketch_tensor, pad], dim=1)
                        model_batch['agents_future'][0, :, :, : sketch_tensor.shape[2]] = sketch_tensor

                # RUN BATCHED ROLLOUTS
                # We want 'num_samples' rollouts total.
                # We process them in chunks of 'batch_size'.
                
                total_rollouts = num_samples
                all_rollouts_list = []
                all_masks_list = []
                
                gt_future_np = None
                if 'agents_future' in model_batch and isinstance(model_batch['agents_future'], torch.Tensor):
                     gt_future_np = model_batch['agents_future'][0].cpu().numpy()

                processed = 0
                while processed < total_rollouts:
                    current_batch_size = min(batch_size, total_rollouts - processed)
                    
                    # Run batch
                    traj, valid = run_batch_rollouts(
                        model=model,
                        model_name=model_name,
                        initial_batch=model_batch, # Passing original batch, it gets replicated inside
                        args=args,
                        horizon_steps=horizon_steps,
                        total_steps=total_steps,
                        num_iterations=num_iterations,
                        map_bounds=map_bounds,
                        start_seed=args.seed + processed,
                        num_rollouts=current_batch_size,
                        gt_future=gt_future_np,
                        conditioning_latents=conditioning_latents if is_latent_mode else None,
                        original_sketch=original_sketch,
                        is_latent_mode=is_latent_mode,
                        num_candidates=args.num_candidates
                    )
                    
                    all_rollouts_list.append(traj)
                    all_masks_list.append(valid)
                    processed += current_batch_size
                
                # Concatenate results
                all_rollouts_np = np.concatenate(all_rollouts_list, axis=0) # [Total, A, T, 4]
                all_rollouts_valid_np = np.concatenate(all_masks_list, axis=0)

                all_model_trajectories[model_name] = {
                    'rollouts': all_rollouts_np,
                    'valid_mask': all_rollouts_valid_np,
                }

                # Visualization (use first rollout)
                vis_traj = all_rollouts_np[0, :, :, :3]
                vis_traj_xyyaw = np.zeros_like(vis_traj)
                vis_traj_xyyaw[..., 0] = all_rollouts_np[0, ..., 0]
                vis_traj_xyyaw[..., 1] = all_rollouts_np[0, ..., 1]
                vis_traj_xyyaw[..., 2] = all_rollouts_np[0, ..., 3] # heading
                predictions_for_vis.append((model_title, vis_traj_xyyaw))

                if model_name == 'controlnet' and original_sketch is not None:
                    conditioning_futures[model_title] = original_sketch

            # Visualization & Save (Same as before)
            if args.save_simulation:
                vis_path = os.path.join(save_dir, f'{scenario_id}_closedloop_video')
                create_trajectory_video(
                    scenario_data, predictions_for_vis, vis_path,
                    f'{scenario_id} (Closed-Loop)', conditioning_futures, controlled_agent_ids
                )

            result_path = os.path.join(save_dir, f'{scenario_id}_closedloop_wosac.pkl')
            if 'agents_id' in batch: agents_id = batch['agents_id'][0].cpu().numpy()
            else: agents_id = np.arange(batch['agents_history'].shape[1], dtype=np.int32)
            last_history_valid = np.any(batch['agents_history'][0].cpu().numpy()[:, -1, :2] != 0, axis=-1)

            with open(result_path, 'wb') as f:
                pickle.dump({
                    'scenario_id': scenario_id,
                    'model_rollouts': all_model_trajectories,
                    'agents_id': agents_id,
                    'sim_agent_mask': last_history_valid,
                    'num_agents': len(agents_id),
                    'num_rollouts': num_samples,
                    'num_steps': total_steps,
                    'horizon_steps': horizon_steps,
                    'total_steps': total_steps,
                    'num_iterations': num_iterations,
                    'sketch': original_sketch,
                    'controlled_agents': controlled_agent_ids,
                    'is_latent_mode': is_latent_mode,
                    'map_bounds': map_bounds,
                    'seed': args.seed,
                    'test_mode': args.test_mode,
                }, f)

        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            import traceback
            traceback.print_exc()
            continue

    print("Closed-loop simulation COMPLETE!")
    print(f"{'='*70}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, required=True, help='Directory containing processed PKL files')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_mode', type=str, default='diffusion', choices=['diffusion', 'prior'])
    parser.add_argument('--save_simulation', action='store_true')
    parser.add_argument(
        '--plot_control_samples',
        action='store_true',
        help='When set, sample 32 ControlNet trajectories and save a combined plot per scenario',
    )
    parser.add_argument('--use_training_format', action='store_true', help='Must be set; TFRecord format is not supported')
    parser.add_argument('--max_scenarios', type=int, default=-1, help='Number of valid scenarios to evaluate (-1 for all)')
    parser.add_argument('--ctrl_ckpt_path', type=str, default=None, help='Optional ControlNet checkpoint for side-by-side comparison')
    parser.add_argument(
        '--only_controlnet',
        action='store_true',
        help='Only run ControlNet (skip baseline) for speed.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_closedloop_1208',
        help='Output subdirectory name inside testing_results (default: test_closedloop_1208)',
    )

    # Closed-loop specific arguments
    parser.add_argument('--closedloop', action='store_true', help='Enable closed-loop simulation with receding horizon')
    parser.add_argument('--horizon_steps', type=int, default=11, help='Steps to execute per iteration (11 = 1.1s @ 10Hz)')
    parser.add_argument('--total_steps', type=int, default=80, help='Total steps to simulate (80 = 8s @ 10Hz)')
    parser.add_argument('--batch_size', type=int, default=2
    , help='Batch size for parallel rollouts (reduce if OOM)')
    parser.add_argument('--num_samples', type=int, default=32, help='Total number of rollout samples per scenario')
    parser.add_argument('--num_candidates', type=int, default=4, help='Number of candidate trajectories to sample per step for selection')

    return parser.parse_args()


def parse_args_latent() -> argparse.Namespace:
    """Parse arguments for latent conditioning inference (both open-loop and closed-loop)."""
    parser = argparse.ArgumentParser(description='VBD Latent Conditioning Inference')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, required=True, help='Directory containing VBD PKL files')
    parser.add_argument('--latent_path', type=str, required=True, help='Directory containing latent PKL files')
    parser.add_argument('--model_path', type=str, required=True, help='Base VBD model checkpoint')
    parser.add_argument('--ctrl_ckpt_path', type=str, required=True, help='ControlNet checkpoint')
    parser.add_argument(
        '--only_controlnet',
        action='store_true',
        help='Only run ControlNet (skip baseline) for speed.',
    )
    parser.add_argument('--test_mode', type=str, default='diffusion', choices=['diffusion', 'prior'])
    parser.add_argument('--save_simulation', action='store_true')
    parser.add_argument('--plot_control_samples', action='store_true', help='Plot 32 ControlNet samples per scenario')
    parser.add_argument('--use_training_format', action='store_true', default=True, help='Must be set; TFRecord format is not supported')
    parser.add_argument('--max_scenarios', type=int, default=-1, help='Number of scenarios to evaluate (-1 for all)')

    # Latent transfer parameters
    parser.add_argument('--source_scenario', type=str, required=True, help='Source scenario ID (without .pkl suffix)')
    parser.add_argument('--source_agents', type=str, required=True, help='Comma-separated source agent IDs (e.g., "0,1,2")')
    parser.add_argument('--target_agents', type=str, required=True, help='Comma-separated target agent IDs (e.g., "1,3,5")')

    # Closed-loop specific arguments
    parser.add_argument('--closedloop', action='store_true', help='Enable closed-loop simulation with receding horizon')
    parser.add_argument('--horizon_steps', type=int, default=11, help='Steps to execute per iteration (11 = 1.1s @ 10Hz)')
    parser.add_argument('--total_steps', type=int, default=80, help='Total steps to simulate (80 = 8s @ 10Hz)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for parallel rollouts (reduce if OOM)')
    parser.add_argument('--num_samples', type=int, default=32, help='Total number of rollout samples per scenario')
    parser.add_argument('--num_candidates', type=int, default=4, help='Number of candidate trajectories to sample per step for selection')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_closedloop_1208',
        help='Output subdirectory name inside testing_results (default: test_closedloop_1208)',
    )

    args = parser.parse_args()

    # Parse comma-separated agent lists
    args.source_agents = [int(x.strip()) for x in args.source_agents.split(',')]
    args.target_agents = [int(x.strip()) for x in args.target_agents.split(',')]

    if len(args.source_agents) != len(args.target_agents):
        raise ValueError(
            f"Number of source agents ({len(args.source_agents)}) must match "
            f"number of target agents ({len(args.target_agents)})"
        )

    return args
