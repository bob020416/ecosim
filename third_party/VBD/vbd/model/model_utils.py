import torch
import numpy as np


def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, x].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.

    """
    x = trajs[..., 0]
    y = trajs[..., 1]
    theta = trajs[..., 2]
    v_x = trajs[..., 3]
    v_y = trajs[..., 4]
    
    local_x = (x - x[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None])
    local_y = -(x - x[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None]) + \
        (y - y[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None])
    
    local_theta = theta - theta[:, :, ref_idx, None]
    local_theta = wrap_angle(local_theta)

    local_v_x = v_x * torch.cos(theta[:, :, ref_idx, None]) + v_y * torch.sin(theta[:, :, ref_idx, None])
    local_v_y = -v_x * torch.sin(theta[:, :, ref_idx, None]) + v_y * torch.cos(theta[:, :, ref_idx, None])

    local_trajs = torch.stack([local_x, local_y, local_theta, local_v_x, local_v_y], dim=-1)
    local_trajs[trajs[..., :5] == 0] = 0

    if trajs.shape[-1] > 5:
        trajs = torch.cat([local_trajs, trajs[..., 5:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs


def batch_transform_polylines_to_local_frame(polylines):
    """
    Batch transform polylines to the local frame of reference.

    Args:
        polylines (torch.Tensor): Polylines tensor of shape [B, M, W, 5].

    Returns:
        torch.Tensor: Transformed polylines in the local frame.

    """
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]
    
    local_x = (x - x[:, :, 0, None]) * torch.cos(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.sin(theta[:, :, 0, None])
    local_y = -(x - x[:, :, 0, None]) * torch.sin(theta[:, :, 0, None]) + \
        (y - y[:, :, 0, None]) * torch.cos(theta[:, :, 0, None])
    
    local_theta = theta - theta[:, :, 0, None]
    local_theta = wrap_angle(local_theta)

    local_polylines = torch.stack([local_x, local_y, local_theta], dim=-1)
    local_polylines[polylines[..., :3] == 0] = 0
    polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)

    return polylines


def batch_transform_trajs_to_global_frame(trajs, current_states):
    """
    Batch transform trajectories to the global frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, x, 2 or 3 or 5].
        current_states (torch.Tensor): Current states tensor of shape [B, N, 5].

    Returns:
        torch.Tensor: Transformed trajectories in the global frame. [B, N, x, 2/3/5]
                     Returns same number of features as input (preserves v_x, v_y if present)

    """
    x, y, theta = current_states[:, :, 0], current_states[:, :, 1], current_states[:, :, 2]
    g_x = trajs[..., 0] * torch.cos(theta[:, :, None]) - trajs[..., 1] * torch.sin(theta[:, :,  None])
    g_y = trajs[..., 0] * torch.sin(theta[:, :, None]) + trajs[..., 1] * torch.cos(theta[:, :,  None])
    x = g_x + x[:, :, None]
    y = g_y + y[:, :, None]

    if trajs.shape[-1] == 2:
        trajs = torch.stack([x, y], dim=-1)
    elif trajs.shape[-1] == 3:
        theta = trajs[..., 2] + theta[:, :, None]
        theta = wrap_angle(theta)
        trajs = torch.stack([x, y, theta], dim=-1)
    else:  # trajs.shape[-1] >= 5, preserve all features including velocities
        theta_transformed = trajs[..., 2] + theta[:, :, None]
        theta_transformed = wrap_angle(theta_transformed)
        # Preserve v_x, v_y and any additional features
        trajs = torch.cat([
            torch.stack([x, y, theta_transformed], dim=-1),
            trajs[..., 3:]  # Keep v_x, v_y and any other features
        ], dim=-1)

    return trajs


def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle tensor.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    # return torch.atan2(torch.sin(angle), torch.cos(angle))
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def inverse_kinematics(
    agents_future: torch.Tensor,
    agents_future_valid: torch.Tensor,
    dt: float = 0.1,
    action_len: int = 5,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future (torch.Tensor): Future agent positions tensor. 
            [B, A, T, 8] # x, y, yaw, velx, vely, length, width, height 
        agents_future_valid (torch.Tensor): Future agent validity tensor. [B, A, T]
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 5.

    Returns:
        torch.Tensor: Predicted actions.

    """
    # Inverse kinematics implementation goes here
    batch_size, num_agents, num_timesteps, _ = agents_future.shape
    assert (num_timesteps-1) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (num_timesteps-1) // action_len
    
    yaw = agents_future[..., 2]
    speed = torch.norm(agents_future[..., 3:5], dim=-1)
    
    yaw_rate = wrap_angle(torch.diff(yaw, dim=-1)) / dt
    accel = torch.diff(speed, dim=-1) / dt
    action_valid = agents_future_valid[..., :1] & agents_future_valid[..., 1:]
    
    # filter out invalid actions
    yaw_rate = torch.where(action_valid, yaw_rate, 0.0)
    accel = torch.where(action_valid, accel, 0.0)
    
    # Reshape for mean pooling
    yaw_rate = yaw_rate.reshape(batch_size, num_agents, num_actions, -1)
    accel = accel.reshape(batch_size, num_agents, num_actions, -1)
    action_valid = action_valid.reshape(batch_size, num_agents, num_actions, -1)
    
    yaw_rate_sample = yaw_rate.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    accel_sample = accel.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    action = torch.stack([accel_sample, yaw_rate_sample], dim=-1)
    action_valid = action_valid.any(dim=-1)
    
    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)
    
    return action, action_valid


def roll_out(
        current_states: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
        global_frame: bool = True,
        add_noise: bool = True
    ):
        """
        Forward pass of the dynamics model.

        Args:
            current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
            actions (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [Accel, yaw_rate]
            global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.
            add_noise (bool): Whether to add stochastic noise during rollout. Default is True (backward compatible).
                            Set to False for closed-loop training to avoid gradient instability and error accumulation.

        Returns:
            torch.Tensor: Predicted trajectories.

        """
        x = current_states[..., 0]
        y = current_states[..., 1]
        theta = current_states[..., 2]
        v_x = current_states[..., 3]
        v_y = current_states[..., 4]
        v = torch.sqrt(v_x**2 + v_y**2)

        a = actions[..., 0].repeat_interleave(action_len, dim=-1)
        v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1)

        # FIXED: Add noise only when add_noise=True (default for backward compatibility)
        # For closed-loop training, this should be set to False to avoid:
        # 1. Gradient instability (stochastic gradients)
        # 2. Error accumulation across receding horizon windows
        # 3. Loss not reflecting true model performance
        if add_noise:
            v = v + torch.randn_like(v) * 0.1
        v = torch.clamp(v, min=0)

        yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1)
        if add_noise:
            yaw_rate = yaw_rate + torch.randn_like(yaw_rate) * 0.01

        if global_frame:
            theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
        else:
            theta = torch.cumsum(yaw_rate * dt, dim=2)

        # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
        # theta = wrap_angle(theta)
        
        v_x = v * torch.cos(theta)
        v_y = v * torch.sin(theta)
        
        if global_frame:
            x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
            y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
        else:
            x = torch.cumsum(v_x * dt, dim=-1)
            y = torch.cumsum(v_y * dt, dim=-1)

        return torch.stack([x, y, theta, v_x, v_y], dim=-1)


# ============================================================================
# Closed-loop Training Utilities
# ============================================================================

def calculate_relations_torch(
    agents_history: torch.Tensor,  # [B, A, H, 8]
    polylines: torch.Tensor,        # [B, P, N, 5]
    traffic_lights: torch.Tensor,   # [B, TL, 3]
) -> torch.Tensor:
    """
    Torch version of calculate_relations (from data_utils.py)
    Preserves gradients for backpropagation in closed-loop training.

    Computes pairwise relations (local position and orientation) between
    agents, polylines, and traffic lights based on the last frame of agents_history.

    Args:
        agents_history: Agent trajectories [B, A, H, 8] - (x, y, heading, vx, vy, length, width, height)
        polylines: Road network [B, P, N, 5] - (x, y, heading, traffic_light_state, lane_type)
        traffic_lights: Traffic light positions [B, TL, 3] - (x, y, state)

    Returns:
        relations: [B, N_total, N_total, 3] where N_total = A + P + TL
                   Each element is (local_pos_x, local_pos_y, theta_diff)
    """
    B = agents_history.shape[0]
    device = agents_history.device

    all_relations = []

    for b in range(B):
        agents = agents_history[b]     # [A, H, 8]
        polys = polylines[b]           # [P, N, 5]
        tls = traffic_lights[b]        # [TL, 3]

        n_agents = agents.shape[0]
        n_polylines = polys.shape[0]
        n_traffic_lights = tls.shape[0]
        n = n_agents + n_polylines + n_traffic_lights

        # Concatenate all elements
        # Agents: last frame [A, 3] - (x, y, heading)
        # Polylines: first point [P, 3] - (x, y, heading)
        # Traffic lights: FIXED - use state instead of dummy zero [TL, 3] - (x, y, state)
        if tls.shape[1] >= 3:
            tls_3d = tls[:, :3]  # [TL, 3] - (x, y, state)
        else:
            # Fallback for old data without state
            tls_3d = torch.cat([tls[:, :2], torch.zeros(n_traffic_lights, 1, device=device)], dim=1)

        all_elements = torch.cat([
            agents[:, -1, :3],  # [A, 3]
            polys[:, 0, :3],    # [P, 3]
            tls_3d              # [TL, 3]
        ], dim=0)  # [N, 3]

        # Compute pairwise position differences using broadcasting
        # [N, 1, 2] - [1, N, 2] = [N, N, 2]
        pos_diff = all_elements[:, :2].unsqueeze(1) - all_elements[:, :2].unsqueeze(0)

        # Transform to local frame
        cos_theta = torch.cos(all_elements[:, 2]).unsqueeze(1)  # [N, 1]
        sin_theta = torch.sin(all_elements[:, 2]).unsqueeze(1)  # [N, 1]
        local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
        local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta

        # Compute angle differences
        theta_diff = all_elements[:, 2].unsqueeze(1) - all_elements[:, 2].unsqueeze(0)  # [N, N]
        theta_diff = torch.atan2(torch.sin(theta_diff), torch.cos(theta_diff))  # wrap to [-pi, pi]

        # Set theta_diff to zero for traffic lights (no orientation)
        start_idx = n_agents + n_polylines
        tl_mask = (torch.arange(n, device=device) >= start_idx).unsqueeze(1) | \
                  (torch.arange(n, device=device) >= start_idx).unsqueeze(0)
        theta_diff = torch.where(tl_mask, torch.zeros_like(theta_diff), theta_diff)

        # Set diagonal to small epsilon value
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        epsilon = 0.01
        local_pos_x = torch.where(diag_mask, torch.full_like(local_pos_x, epsilon), local_pos_x)
        local_pos_y = torch.where(diag_mask, torch.full_like(local_pos_y, epsilon), local_pos_y)
        theta_diff = torch.where(diag_mask, torch.full_like(theta_diff, epsilon), theta_diff)

        # Mask invalid elements (zero coordinates)
        zero_mask = (all_elements[:, 0].unsqueeze(1) == 0) | (all_elements[:, 0].unsqueeze(0) == 0)

        # Stack relations
        relations = torch.stack([local_pos_x, local_pos_y, theta_diff], dim=-1)  # [N, N, 3]
        relations = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(relations), relations)

        all_relations.append(relations)

    return torch.stack(all_relations, dim=0)  # [B, N, N, 3]


def update_batch_for_training(
    batch: dict[str, torch.Tensor],
    predicted_segment: torch.Tensor,  # [B, A, segment_len, 5] - (x, y, heading, vx, vy)
) -> dict[str, torch.Tensor]:
    """
    Update batch for closed-loop training (Torch version of update_batch_with_prediction)
    Preserves gradients for backpropagation.

    Updates:
    1. agents_history: Shift and append new predicted states
    2. relations: Recompute based on new agents_history

    Args:
        batch: Current batch dict
        predicted_segment: Predicted trajectory segment [B, A, segment_len, 5]
                          where segment_len is the number of timesteps to execute

    Returns:
        Updated batch dict (new dict, doesn't modify input)
    """
    B, A, H, D = batch['agents_history'].shape
    device = batch['agents_history'].device
    segment_steps = predicted_segment.shape[2]  # Infer from predicted_segment

    # 1. Construct new history states [B, A, segment_steps, D]
    # CRITICAL FIX: Build from predicted_segment to preserve gradients!
    # torch.zeros().copy_() would break the gradient chain
    if D == 5:
        # Simple case: just use predicted_segment directly
        new_states = predicted_segment[..., :5]
    else:
        # Need to append vehicle dimensions
        vehicle_dims = batch['agents_history'][:, :, -1:, 5:].expand(-1, -1, segment_steps, -1)
        # Use torch.cat to preserve gradients from predicted_segment
        new_states = torch.cat([
            predicted_segment[..., :5],  # Has gradients
            vehicle_dims                  # Static, no grad needed
        ], dim=-1)

    # 2. Shift history: remove oldest, append newest
    updated_history = torch.cat([
        batch['agents_history'][:, :, segment_steps:, :],  # Keep newer frames
        new_states                                          # Append predicted
    ], dim=2)

    # 3. Ensure history length doesn't exceed H (align with inference)
    # This is critical when segment_steps > H (e.g., segment_len=27, H=11)
    if updated_history.shape[2] > H:
        # Take only the last H frames to maintain consistent history length
        updated_history = updated_history[:, :, -H:, :]
    elif updated_history.shape[2] < H:
        # Pad with zeros if needed (should rarely happen)
        pad_len = H - updated_history.shape[2]
        pad = torch.zeros(B, A, pad_len, D, device=device)
        updated_history = torch.cat([pad, updated_history], dim=2)

    # 4. Recompute relations (Torch version)
    updated_relations = calculate_relations_torch(
        updated_history,
        batch['polylines'],
        batch['traffic_light_points']
    )

    # 5. Create updated batch (don't modify original)
    updated_batch = {k: v for k, v in batch.items()}
    updated_batch['agents_history'] = updated_history
    updated_batch['relations'] = updated_relations

    return updated_batch