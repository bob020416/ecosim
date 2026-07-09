# Ctrl VBD Open-loop with Testing with custom sketch and Video Visualization
import argparse
import csv
import glob
import math
import os
import pickle
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
        control_agents_future: torch.Tensor | None = None,
        control_agent_latents: torch.Tensor | None = None,
        control_scenario_latents: torch.Tensor | None = None,
        control_text_prompt=None,
        control_text_mask: torch.Tensor | None = None,
        control_mask: torch.Tensor | None = None,
    ):
        """
        Single diffusion step to obtain x_{t-1} given x_t and encoder outputs.
        Supports both sketch (agents_future) and latent (agent_latents + scenario_latents) conditioning.
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
        encoder_outputs = self.encoder(batch)
        if num_samples > 1:
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
        iterator = tqdm(diffusion_steps, total=len(diffusion_steps), desc="Diffusion") if use_tqdm else diffusion_steps
        for t in iterator:
            x_t_history.append(x_t.detach().cpu().numpy())
            denoiser_outputs, x_t = self.step_denoiser(
                x_t=x_t,
                c=encoder_outputs,
                t=fix_t if fix_t >= 0 else t,
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
            denoiser_output_history.append(torch_dict_to_numpy(denoiser_outputs))
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

    keys = [
        'agents_history', 'agents_interested', 'agents_type', 'agents_future',
        'traffic_light_points', 'polylines', 'polylines_valid', 'relations'
    ]

    for key in keys:
        if key not in pkl_data:
            continue

        data = pkl_data[key]

        if key in {'agents_history', 'agents_interested', 'agents_type', 'agents_future'}:
            data = data[:n_agents]

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
        for tl in traffic_lights:
            if tl[0] == 0:
                continue
            if (
                np.abs(tl[0]) < COORD_LIMIT
                and np.abs(tl[1]) < COORD_LIMIT
            ):
                ax.add_patch(plt.Circle((tl[0], tl[1]), 1.0, color='red', alpha=0.5))

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
    # This ensures camera follows actual predicted trajectory
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
# Metrics
# -----------------------------------------------------------------------------
def compute_openloop_metrics(
    gt_future: np.ndarray,
    pred_traj: np.ndarray,
    interested_mask: np.ndarray,
) -> Tuple[float, float, int, int]:
    """Return ADE, FDE, number of valid agents, and number of skipped agents."""
    n_agents = min(gt_future.shape[0], pred_traj.shape[0], N_SIM_AGENTS)
    n_timesteps = min(pred_traj.shape[1], gt_future.shape[1] - 1, 80)

    gt_pos = gt_future[:n_agents, 1 : 1 + n_timesteps, :2]
    pred_pos = pred_traj[:n_agents, :n_timesteps, :2]
    mask = interested_mask[:n_agents]

    ade_per_agent: List[float] = []
    fde_per_agent: List[float] = []
    skipped_agents = 0

    for i in range(n_agents):
        if not mask[i]:
            continue

        gt_valid = (
            (gt_pos[i, :, 0] != 0)
            & (gt_pos[i, :, 0] != -1)
            & (np.abs(gt_pos[i, :, 0]) < COORD_LIMIT)
            & (np.abs(gt_pos[i, :, 1]) < COORD_LIMIT)
        )
        pred_valid = (
            (pred_pos[i, :, 0] != 0)
            & (pred_pos[i, :, 0] != -1)
            & (np.abs(pred_pos[i, :, 0]) < COORD_LIMIT)
            & (np.abs(pred_pos[i, :, 1]) < COORD_LIMIT)
        )

        both_valid = gt_valid & pred_valid
        if not np.any(both_valid):
            skipped_agents += 1
            continue

        valid_gt = gt_pos[i, both_valid]
        valid_pred = pred_pos[i, both_valid]
        distances = np.linalg.norm(valid_gt - valid_pred, axis=1)
        ade_per_agent.append(float(np.mean(distances)))

        last_valid_idx = np.where(both_valid)[0][-1]
        final_error = np.linalg.norm(gt_pos[i, last_valid_idx] - pred_pos[i, last_valid_idx])
        fde_per_agent.append(float(final_error))

    ade = float(np.mean(ade_per_agent)) if ade_per_agent else float('nan')
    fde = float(np.mean(fde_per_agent)) if fde_per_agent else float('nan')
    return ade, fde, len(ade_per_agent), skipped_agents

# -----------------------------------------------------------------------------
# Evaluation driver
# -----------------------------------------------------------------------------




def run_simulation(
    args: argparse.Namespace,
    sketch_fn: SketchFn,
    sketch_name: str = 'Sketch',
    batch_preprocess_fn = None,
) -> None:
    if args.model_path is None:
        raise ValueError("--model_path must be provided")
    if args.test_path is None:
        raise ValueError("--test_path must be provided")
    if not args.use_training_format:
        raise NotImplementedError(
            "TFRecord closed-loop testing is no longer supported; "
            "please export PKL files and pass --use_training_format."
        )

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

    set_seed(args.seed)

    sketch_slug = (
        sketch_name.lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('°', 'deg')
    )

    model_specs: List[Tuple[str, torch.nn.Module, str]] = [
        ('baseline', base_model, 'Baseline')
    ]
    if control_model is not None:
        model_specs.append(('controlnet', control_model, 'ControlNet'))

    pkl_files = load_pkl_files(args.test_path)
    if not pkl_files:
        raise RuntimeError(f"No PKL files found in {args.test_path}")

    if args.max_scenarios > 0:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(pkl_files))
    else:
        order = np.arange(len(pkl_files))

    save_dir = os.path.join('testing_results', f'test_{args.test_mode}', str(args.seed))
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    header = ['scenario_id']
    for name, _, _ in model_specs:
        header.extend([
            f'ADE_{name}',
            f'FDE_{name}',
            f'num_valid_{name}',
            f'num_skipped_{name}',
        ])
    header.extend([
        'ADE_sketch',
        'FDE_sketch',
        'num_valid_sketch',
        'num_skipped_sketch',
    ])

    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(header)

    selected_files: List[str] = []
    processed_valid = 0

    for idx in order:
        scenario_path = pkl_files[idx]
        scenario_id = os.path.basename(scenario_path).replace('.pkl', '')
        print(f"Running scenario {scenario_id}...")

        try:
            scenario_raw = load_pkl_data(scenario_path) 
            scenario_gt = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in scenario_raw.items() # 複製一份 for gt 
            }
            batch = prepare_model_batch(scenario_raw)
            if batch_preprocess_fn is not None:
                batch_preprocess_fn(batch, scenario_raw)
            conditioning_output, controlled_agent_ids = sketch_fn(batch)
            if controlled_agent_ids is None:
                controlled_agent_ids = []

            # Determine conditioning mode: sketch (np.ndarray) or latent (dict)
            is_latent_mode = isinstance(conditioning_output, dict) and 'agent_latents' in conditioning_output
            conditioning_future_np = None
            conditioning_latents = None

            if is_latent_mode:
                # Latent conditioning mode
                conditioning_latents = conditioning_output
            else:
                # Sketch conditioning mode (original behavior)
                conditioning_future_np = conditioning_output

            conditioning_futures: dict[str, np.ndarray] = {}
            if conditioning_future_np is not None and isinstance(conditioning_future_np, np.ndarray):
                conditioning_futures['ControlNet'] = conditioning_future_np.copy()

            # Clone every tensor so baseline and ControlNet operate on independent inputs.
            # `batch` remains our mutable source (with the sketch); `base_batch` holds detached
            # copies that each model will clone again before inference.
            base_batch = {
                key: value.clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }

            metrics_row = [scenario_id]
            predictions_for_vis: List[Tuple[str, np.ndarray]] = []
            baseline_valid = False
            controlnet_present = False

            gt_future = scenario_gt['agents_future']
            interested_mask = scenario_gt['agents_interested'] > 0

            sketch_metrics: dict[str, Tuple[float, float, int, int]] = {}

            for name, model, title in model_specs:
                # Each model receives its own clone to guard against in-place ops performed
                # during forward passes (e.g. Lightning moving tensors to device).
                model_inputs = {
                    key: value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in base_batch.items()
                }

                # Add latent conditioning for controlnet if in latent mode
                if name == 'controlnet' and is_latent_mode and conditioning_latents is not None:
                    model_inputs['agent_latents'] = torch.from_numpy(
                        conditioning_latents['agent_latents']
                    ).unsqueeze(0).to(args.device)
                    model_inputs['scenario_latents'] = torch.from_numpy(
                        conditioning_latents['scenario_latents']
                    ).unsqueeze(0).to(args.device)

                with torch.no_grad():
                    if args.test_mode == 'diffusion':
                        pred = model.sample_denoiser(model_inputs)
                        pred_traj = pred['denoised_trajs'].cpu().numpy()[0]
                    elif args.test_mode == 'prior':
                        pred = model.inference_predictor(model_inputs)
                        scores = pred['goal_scores'][0].softmax(dim=-1)
                        trajs = pred['goal_trajs'][0]
                        sampled_idx = torch.multinomial(scores, 1).squeeze(-1)
                        agent_idx = torch.arange(trajs.shape[0], device=sampled_idx.device)
                        pred_traj = trajs[agent_idx, sampled_idx].cpu().numpy()
                    else:
                        raise NotImplementedError(f"Unsupported test_mode: {args.test_mode}")

                ade, fde, num_valid, num_skipped = compute_openloop_metrics(
                    gt_future, pred_traj, interested_mask
                )

                if name == 'baseline' and num_valid > 0:
                    baseline_valid = True

                metrics_row.extend([ade, fde, num_valid, num_skipped])

                if num_valid == 0:
                    print(
                        f"  ⚠️ {title}: no overlapping valid timesteps; metrics reported as NaN."
                    )
                else:
                    print(
                        f"  {title}: ADE={ade:.3f}, FDE={fde:.3f}, valid={num_valid}, skipped={num_skipped}"
                    )

                predictions_for_vis.append((title, pred_traj))

                if name == 'controlnet':
                    controlnet_present = True
                    # Compute conditioning alignment metrics (sketch or latent mode)
                    if controlled_agent_ids:
                        mask_for_conditioning = np.zeros_like(interested_mask, dtype=bool)
                        mask_for_conditioning[controlled_agent_ids] = True
                        if conditioning_future_np is not None:
                            # Sketch mode: compare pred to sketch
                            sketch_metrics['controlnet'] = compute_openloop_metrics(
                                conditioning_future_np,
                                pred_traj,
                                mask_for_conditioning,
                            )
                        elif is_latent_mode:
                            # Latent mode: compare pred to GT for conditioned agents
                            sketch_metrics['controlnet'] = compute_openloop_metrics(
                                gt_future,
                                pred_traj,
                                mask_for_conditioning,
                            )
                elif name == 'baseline':
                    if conditioning_future_np is not None and controlled_agent_ids:
                        mask_for_conditioning = np.zeros_like(interested_mask, dtype=bool)
                        mask_for_conditioning[controlled_agent_ids] = True
                        sketch_metrics['baseline'] = compute_openloop_metrics(
                            conditioning_future_np,
                            pred_traj,
                            mask_for_conditioning,
                        )
                    elif is_latent_mode and controlled_agent_ids:
                        mask_for_conditioning = np.zeros_like(interested_mask, dtype=bool)
                        mask_for_conditioning[controlled_agent_ids] = True
                        sketch_metrics['baseline'] = compute_openloop_metrics(
                            gt_future,
                            pred_traj,
                            mask_for_conditioning,
                        )

                if name == 'controlnet' and 'ControlNet' in conditioning_futures:
                    conditioning_futures[title] = conditioning_futures.pop('ControlNet')


            if 'controlnet' in sketch_metrics:
                ade_sketch, fde_sketch, valid_sketch, skipped_sketch = sketch_metrics['controlnet']
            elif 'baseline' in sketch_metrics:
                ade_sketch, fde_sketch, valid_sketch, skipped_sketch = sketch_metrics['baseline']
            else:
                ade_sketch, fde_sketch, valid_sketch, skipped_sketch = (float('nan'),) * 4

            metrics_row.extend([ade_sketch, fde_sketch, valid_sketch, skipped_sketch])
            if not np.isnan(ade_sketch):
                print(
                    f"  {sketch_name} alignment: ADE={ade_sketch:.3f}, FDE={fde_sketch:.3f}, "
                    f"valid={valid_sketch}, skipped={skipped_sketch}"
                )

            if not baseline_valid:
                print("  ⚠️ Baseline invalid for this scenario; skipping.")
                continue

            with open(metrics_path, 'a', newline='') as f:
                csv.writer(f).writerow(metrics_row)

            selected_files.append(scenario_path)
            processed_valid += 1

            if args.save_simulation:
                vis_path = os.path.join(save_dir, f'{scenario_id}_video')
                create_trajectory_video(
                    scenario_gt,
                    predictions_for_vis,
                    vis_path,
                    scenario_id,
                    conditioning_futures=(
                        conditioning_futures
                        if controlnet_present and conditioning_futures
                        else None
                    ),
                    controlled_agent_ids=controlled_agent_ids if controlnet_present else None,
                )

            if controlnet_present and args.plot_control_samples:
                try:
                    sketch_future = conditioning_future_np
                    if sketch_future is None and 'agents_future' in base_batch:
                        sketch_future = base_batch['agents_future'][0].detach().cpu().numpy()
                    ids_for_plot = controlled_agent_ids
                    if (not ids_for_plot) and sketch_future is not None:
                        ids_for_plot = extract_controlled_agent_indices(sketch_future)
                    if sketch_future is not None and ids_for_plot:
                        samples: List[np.ndarray] = []
                        for _ in range(32):
                            sample_inputs = {
                                key: value.clone()
                                if isinstance(value, torch.Tensor)
                                else value
                                for key, value in base_batch.items()
                            }
                            single_sample = control_model.sample_denoiser(
                                sample_inputs,
                                num_samples=1,
                                use_tqdm=False,
                            )
                            denoised = single_sample.get('denoised_trajs')
                            if denoised is None:
                                raise ValueError('ControlNet output missing denoised_trajs')
                            if isinstance(denoised, torch.Tensor):
                                traj = denoised.detach().cpu().numpy()
                            else:
                                traj = np.asarray(denoised)
                            samples.append(traj[0])
                        control_samples = np.stack(samples, axis=0)
                        plot_path = os.path.join(
                            save_dir,
                            f'{scenario_id}_{sketch_slug}_control_samples.png',
                        )
                        polylines = scenario_gt.get('polylines', np.zeros((0,)))
                        traffic = scenario_gt.get('traffic_light_points', np.zeros((0,)))
                        plot_controlled_trajectories(
                            scenario_gt['agents_history'],
                            control_samples,
                            ids_for_plot,
                            polylines,
                            traffic,
                            sketch_future,
                            sketch_name,
                            plot_path,
                        )
                except Exception as exc:  # pragma: no cover - keep evaluation running
                    print(f"  ⚠️ Failed to plot control samples: {exc}")

            if args.max_scenarios > 0 and processed_valid >= args.max_scenarios:
                break

        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"Error processing {scenario_id}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    if args.max_scenarios > 0 and processed_valid < args.max_scenarios:
        print(
            f"⚠️ Requested {args.max_scenarios} scenarios but only {processed_valid} valid ones were found."
        )

    if selected_files:
        print(
            "Selected scenarios (seed {}): {}".format(
                args.seed,
                [os.path.basename(f) for f in selected_files],
            )
        )


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
    return parser.parse_args()


def parse_args_latent():
    """Parse arguments for latent conditioning inference."""
    parser = argparse.ArgumentParser(description='VBD Latent Conditioning Inference')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, required=True, help='Directory containing VBD PKL files')
    parser.add_argument('--latent_path', type=str, required=True, help='Directory containing latent PKL files')
    parser.add_argument('--model_path', type=str, required=True, help='Base VBD model checkpoint')
    parser.add_argument('--ctrl_ckpt_path', type=str, required=True, help='ControlNet checkpoint')
    parser.add_argument('--test_mode', type=str, default='diffusion', choices=['diffusion', 'prior'])
    parser.add_argument('--save_simulation', action='store_true')
    parser.add_argument('--plot_control_samples', action='store_true', help='Plot 32 ControlNet samples per scenario')
    parser.add_argument('--use_training_format', action='store_true', default=True)
    parser.add_argument('--max_scenarios', type=int, default=-1, help='Number of scenarios to evaluate (-1 for all)')

    # Latent transfer parameters
    parser.add_argument('--source_scenario', type=str, required=True, help='Source scenario ID (without .pkl suffix)')
    parser.add_argument('--source_agents', type=str, required=True, help='Comma-separated source agent IDs (e.g., "0,1,2")')
    parser.add_argument('--target_agents', type=str, required=True, help='Comma-separated target agent IDs (e.g., "1,3,5")')

    args = parser.parse_args()

    # Parse comma-separated agent IDs
    args.source_agents = [int(x.strip()) for x in args.source_agents.split(',')]
    args.target_agents = [int(x.strip()) for x in args.target_agents.split(',')]

    # Validate
    if len(args.source_agents) != len(args.target_agents):
        raise ValueError(f"Number of source agents ({len(args.source_agents)}) must match target agents ({len(args.target_agents)})")

    return args


def load_latent_data(latent_path, scenario_id):
    """Load latent data for a given scenario."""
    latent_file = os.path.join(latent_path, f"{scenario_id}_latents.pkl")
    if not os.path.exists(latent_file):
        raise FileNotFoundError(f"Latent file not found: {latent_file}")

    with open(latent_file, 'rb') as f:
        latent_data = pickle.load(f)

    return latent_data
