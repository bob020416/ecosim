from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover
    import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder


def _wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))

def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def _kl_diag_standard_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(q||N(0,1)) for diagonal Gaussians.

    Args:
        mu/log_var: [..., Z]
    Returns:
        kl: [...] (summed over Z)
    """
    return 0.5 * (mu.pow(2) + torch.exp(log_var) - 1.0 - log_var).sum(dim=-1)

def _kl_diag_gaussians(mu_q: torch.Tensor, log_var_q: torch.Tensor, mu_p: torch.Tensor, log_var_p: torch.Tensor) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians. Returns summed KL over last dim."""
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    kl_per_dim = 0.5 * (
        log_var_p
        - log_var_q
        + (var_q + (mu_q - mu_p).pow(2)) / torch.clamp(var_p, min=1e-8)
        - 1.0
    )
    return kl_per_dim.sum(dim=-1)


def _pick_query_agent_index(agents_interested: torch.Tensor) -> torch.Tensor:
    """Pick query agent per batch by max score; [B,A] -> [B]."""
    if agents_interested.dim() != 2:
        raise ValueError(f"agents_interested must be [B,A], got {agents_interested.shape}")
    return torch.argmax(agents_interested.to(torch.float32), dim=-1).to(torch.long)


def _transform_traj_to_query_frame(
    traj_global: torch.Tensor,  # [B,A,T,3] (x,y,theta)
    query_idx: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Query-centric transform using query agent state at t0."""
    if traj_global.dim() != 4 or traj_global.shape[-1] != 3:
        raise ValueError(f"traj_global must be [B,A,T,3], got {traj_global.shape}")
    if query_idx.dim() != 1 or query_idx.shape[0] != traj_global.shape[0]:
        raise ValueError(f"query_idx must be [B], got {query_idx.shape}")

    b = torch.arange(traj_global.shape[0], device=traj_global.device)
    ref = traj_global[b, query_idx, 0]  # [B,3]
    x0, y0, th0 = ref[:, 0], ref[:, 1], ref[:, 2]

    cos_t = torch.cos(th0)
    sin_t = torch.sin(th0)

    dx = traj_global[..., 0] - x0[:, None, None]
    dy = traj_global[..., 1] - y0[:, None, None]
    local_x = dx * cos_t[:, None, None] + dy * sin_t[:, None, None]
    local_y = -dx * sin_t[:, None, None] + dy * cos_t[:, None, None]
    local_th = _wrap_angle(traj_global[..., 2] - th0[:, None, None])
    return torch.stack([local_x, local_y, local_th], dim=-1)


def _transform_traj_to_agent_frame(
    traj_global: torch.Tensor,  # [B,A,T,3] (x,y,theta)
) -> torch.Tensor:
    """Per-agent local transform using each agent's own t0 state."""
    if traj_global.dim() != 4 or traj_global.shape[-1] != 3:
        raise ValueError(f"traj_global must be [B,A,T,3], got {traj_global.shape}")

    ref = traj_global[:, :, 0]  # [B,A,3]
    x0, y0, th0 = ref[..., 0], ref[..., 1], ref[..., 2]

    cos_t = torch.cos(th0)
    sin_t = torch.sin(th0)

    dx = traj_global[..., 0] - x0[:, :, None]
    dy = traj_global[..., 1] - y0[:, :, None]
    local_x = dx * cos_t[:, :, None] + dy * sin_t[:, :, None]
    local_y = -dx * sin_t[:, :, None] + dy * cos_t[:, :, None]
    local_th = _wrap_angle(traj_global[..., 2] - th0[:, :, None])
    return torch.stack([local_x, local_y, local_th], dim=-1)


def _relative_geometry(
    t0_global: torch.Tensor,  # [B,A,3]
    agent_is_valid: torch.Tensor,  # [B,A]
) -> torch.Tensor:
    """Pairwise relative geometry in each agent's local frame (t0)."""
    if t0_global.dim() != 3 or t0_global.shape[-1] != 3:
        raise ValueError(f"t0_global must be [B,A,3], got {t0_global.shape}")
    if agent_is_valid.dim() != 2 or agent_is_valid.shape[0] != t0_global.shape[0]:
        raise ValueError(f"agent_is_valid must be [B,A], got {agent_is_valid.shape}")

    x = t0_global[..., 0]
    y = t0_global[..., 1]
    th = t0_global[..., 2]

    dx = x[:, None, :] - x[:, :, None]
    dy = y[:, None, :] - y[:, :, None]

    cos_i = torch.cos(th)[:, :, None]
    sin_i = torch.sin(th)[:, :, None]
    local_x = dx * cos_i + dy * sin_i
    local_y = -dx * sin_i + dy * cos_i
    dtheta = _wrap_angle(th[:, None, :] - th[:, :, None])

    rel = torch.stack([local_x, local_y, torch.sin(dtheta), torch.cos(dtheta)], dim=-1)
    valid_pair = agent_is_valid[:, :, None] & agent_is_valid[:, None, :]
    return rel * valid_pair.unsqueeze(-1).to(rel.dtype)


def _traj_to_deltas(traj_q: torch.Tensor) -> torch.Tensor:
    """[B,A,T,3] absolute -> deltas relative to t0 (dx,dy,dtheta)."""
    ref = traj_q[:, :, :1, :]  # [B,A,1,3]
    dxdy = traj_q[..., :2] - ref[..., :2]
    dtheta = _wrap_angle(traj_q[..., 2] - ref[..., 2])
    return torch.cat([dxdy, dtheta.unsqueeze(-1)], dim=-1)


def _dtheta_to_sincos(dtheta: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(dtheta), torch.cos(dtheta)], dim=-1)


class _AgentTrajEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,A,T,D]
        b, a, t, d = x.shape
        x = x.reshape(b * a, t, d)
        _, h = self.gru(x)
        feats = self.norm(h[-1])
        return feats.reshape(b, a, self.hidden_dim)


class _AgentInteraction(nn.Module):
    """Self-attention over agents with relative-geometry bias."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        if layers <= 0:
            self.layers = None
            return
        self.layers = nn.ModuleList([_RelAttentionBlock(dim=dim, heads=heads, dropout=dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, rel_geom: torch.Tensor, agent_is_valid: torch.Tensor) -> torch.Tensor:
        if self.layers is None:
            return x
        for layer in self.layers:
            x = layer(x, rel_geom=rel_geom, agent_is_valid=agent_is_valid)
        return x


class _RelGeometryBias(nn.Module):
    def __init__(self, heads: int, hidden_dim: int, rel_dim: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rel_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, heads),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, rel_geom: torch.Tensor) -> torch.Tensor:
        # rel_geom: [B,A,A,rel_dim] -> [B,H,A,A]
        bias = self.mlp(rel_geom)
        bias = self.dropout(bias)
        return bias.permute(0, 3, 1, 2)


class _RelAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        bias_hidden = max(32, dim // 4)
        self.bias = _RelGeometryBias(heads=heads, hidden_dim=bias_hidden, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, rel_geom: torch.Tensor, agent_is_valid: torch.Tensor) -> torch.Tensor:
        b, a, dim = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, a, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, a, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, a, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self.bias(rel_geom)

        if agent_is_valid is not None:
            key_mask = ~agent_is_valid.to(torch.bool)
            has_valid = agent_is_valid.to(torch.bool).any(dim=-1, keepdim=True)
            key_mask = key_mask & has_valid
            attn = attn.masked_fill(key_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, a, dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        out = x + out

        if agent_is_valid is not None:
            out = out * agent_is_valid.to(out.dtype).unsqueeze(-1)

        h2 = self.norm2(out)
        ff = self.ffn(h2)
        out = out + self.ffn_dropout(ff)

        if agent_is_valid is not None:
            out = out * agent_is_valid.to(out.dtype).unsqueeze(-1)

        return out


class _DeltaDecoder(nn.Module):
    """Decode per-agent latent + time to (dx,dy,sin(dth),cos(dth))."""

    def __init__(
        self, future_len: int, z_agent_dim: int, z_scene_dim: int, hidden_dim: int, layers: int, dropout: float
    ):
        super().__init__()
        self.future_len = future_len
        self.z_agent_dim = z_agent_dim
        self.z_scene_dim = z_scene_dim
        self.time = nn.Embedding(future_len, hidden_dim)

        z_total = z_agent_dim + z_scene_dim
        mlp: list[nn.Module] = [nn.Linear(z_total + hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, layers - 1)):
            mlp += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        mlp += [nn.Linear(hidden_dim, 4)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, z_agent: torch.Tensor, z_scene: Optional[torch.Tensor]) -> torch.Tensor:
        # z_agent: [B,A,Za], z_scene: [B,Zs] (optional) -> [B,A,T,4]
        b, a, za = z_agent.shape
        if za != self.z_agent_dim:
            raise ValueError(f"z_agent last dim must be {self.z_agent_dim}, got {za}")

        if self.z_scene_dim > 0:
            if z_scene is None:
                raise ValueError("z_scene must be provided when z_scene_dim > 0")
            if z_scene.dim() != 2 or z_scene.shape[0] != b or z_scene.shape[1] != self.z_scene_dim:
                raise ValueError(f"z_scene must be [B,{self.z_scene_dim}], got {z_scene.shape}")
            z = torch.cat([z_agent, z_scene[:, None, :].expand(b, a, self.z_scene_dim)], dim=-1)
        else:
            z = z_agent

        t_idx = torch.arange(self.future_len, device=z.device)
        t_emb = self.time(t_idx)[None, None, :, :].expand(b, a, self.future_len, -1)
        zt = torch.cat([z[:, :, None, :].expand(b, a, self.future_len, -1), t_emb], dim=-1)
        return self.mlp(zt)


@dataclass
class BehaviorVAEOutputs:
    query_idx: torch.Tensor  # [B]
    agent_is_valid: torch.Tensor  # [B,A]
    state_valid: torch.Tensor  # [B,A,T]
    traj_gt_q: torch.Tensor  # [B,A,T,3] query frame abs
    traj_gt_local: torch.Tensor  # [B,A,T,3] agent local frame abs
    traj_gt_delta: torch.Tensor  # [B,A,T,3] agent-local (dx,dy,dtheta)
    z_agent: torch.Tensor  # [B,A,Za]
    mu: torch.Tensor  # [B,A,Za]
    log_var: torch.Tensor  # [B,A,Za]
    mu_p: torch.Tensor  # [B,A,Za]
    log_var_p: torch.Tensor  # [B,A,Za]
    z_scene: Optional[torch.Tensor]  # [B,Zs]
    mu_scene: Optional[torch.Tensor]  # [B,Zs]
    log_var_scene: Optional[torch.Tensor]  # [B,Zs]
    mu_scene_p: Optional[torch.Tensor]  # [B,Zs]
    log_var_scene_p: Optional[torch.Tensor]  # [B,Zs]
    traj_pred_delta: torch.Tensor  # [B,A,T,4] (dx,dy,sin,cos)


class BehaviorVAE(pl.LightningModule):
    """Simple-yet-effective baseline for stepwise verification.

    - Local-centric: each agent uses its own t0 frame; query frame is for visualization only.
    - Interaction: relative-geometry-biased self-attention over agent tokens.
    - Reconstruction: decode trajectories in each agent's local frame.
    - Deterministic: no KL, no sampling; intended to *prove* 1-sample overfit first.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        self.agents_len = int(cfg.get("agents_len", 64))
        self.future_len = int(cfg.get("future_len", 81))
        self.truncate_after_invalid = bool(cfg.get("truncate_after_invalid", True))

        self.train_context_encoder = bool(cfg.get("train_context_encoder", False))
        self.use_context = bool(cfg.get("use_context", True))
        self.posterior_use_context = bool(cfg.get("posterior_use_context", self.use_context))
        if not self.use_context:
            self.posterior_use_context = False

        self.z_dim = int(cfg.get("z_dim", 256))
        self.z_scene_dim = int(cfg.get("z_scene_dim", 0))
        self.use_z_scene = bool(cfg.get("use_z_scene", self.z_scene_dim > 0))
        if not self.use_z_scene:
            self.z_scene_dim = 0
        self.hidden_dim = int(cfg.get("hidden_dim", 256))

        self.lr = float(cfg.get("lr", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 0.0))

        # Optional VAE mode (stepwise verification)
        self.use_vae = bool(cfg.get("use_vae", False))
        self.latent_sampling = bool(cfg.get("latent_sampling", True))
        self.latent_sampling_warmup_steps = int(cfg.get("latent_sampling_warmup_steps", 0))
        self.kl_weight = float(cfg.get("kl_weight", 0.0))
        self.kl_warmup_steps = int(cfg.get("kl_warmup_steps", 0))
        self.free_nats = float(cfg.get("free_nats", 0.0))
        self.use_conditional_prior = bool(cfg.get("use_conditional_prior", False))

        dropout = float(cfg.get("dropout", 0.0))
        self.context_encoder = Encoder(
            layers=int(cfg.get("encoder_layers", 6)),
            version=str(cfg.get("encoder_version", "v1")),
        )

        # Encode each agent's query-frame delta trajectory (dx,dy,sin(dth),cos(dth)).
        self.traj_enc = _AgentTrajEncoder(input_dim=4, hidden_dim=self.hidden_dim, layers=2, dropout=dropout)
        self.interaction = _AgentInteraction(
            dim=self.hidden_dim,
            layers=int(cfg.get("interaction_layers", 1)),
            heads=int(cfg.get("interaction_heads", 8)),
            dropout=dropout,
        )
        in_z = self.hidden_dim + (256 if self.use_context else 0)
        if self.use_vae:
            # IMPORTANT: do NOT LayerNorm the concatenated (mu, log_var). That would keep log_var ~N(0,1)
            # and prevent the model from shrinking std to near-zero when KL is off (stepwise overfit).
            self.to_feat = nn.LayerNorm(in_z)
            self.to_mu = nn.Linear(in_z, self.z_dim)
            self.to_logvar = nn.Linear(in_z, self.z_dim)
            nn.init.constant_(self.to_logvar.bias, -4.0)  # start near-deterministic

            # Conditional prior p(z|context): per-agent
            if self.use_conditional_prior:
                if not self.use_context:
                    raise ValueError("use_conditional_prior=True requires use_context=True")
                self.prior_feat = nn.LayerNorm(256)
                self.prior_mu = nn.Linear(256, self.z_dim)
                self.prior_logvar = nn.Linear(256, self.z_dim)
                nn.init.constant_(self.prior_logvar.bias, -2.0)

            # Optional global scene latent
            if self.use_z_scene:
                self.scene_post = nn.Sequential(
                    nn.LayerNorm(in_z),
                    nn.Linear(in_z, self.z_scene_dim * 2),
                )
                self.scene_prior = None
                if self.use_conditional_prior:
                    self.scene_prior = nn.Sequential(
                        nn.LayerNorm(256),
                        nn.Linear(256, self.z_scene_dim * 2),
                    )
        else:
            self.to_z = nn.Sequential(nn.Linear(in_z, self.z_dim), nn.LayerNorm(self.z_dim))
        self.decoder = _DeltaDecoder(
            future_len=self.future_len,
            z_agent_dim=self.z_dim,
            z_scene_dim=self.z_scene_dim,
            hidden_dim=self.hidden_dim,
            layers=int(cfg.get("decoder_mlp_layers", 2)),
            dropout=dropout,
        )

    def configure_optimizers(self):
        params: list[nn.Parameter] = []
        if self.train_context_encoder:
            params += list(self.context_encoder.parameters())
        params += list(self.traj_enc.parameters())
        params += list(self.interaction.parameters())
        if self.use_vae:
            params += list(self.to_feat.parameters())
            params += list(self.to_mu.parameters())
            params += list(self.to_logvar.parameters())
            if self.use_conditional_prior:
                params += list(self.prior_feat.parameters())
                params += list(self.prior_mu.parameters())
                params += list(self.prior_logvar.parameters())
            if self.use_z_scene:
                params += list(self.scene_post.parameters())
                if self.scene_prior is not None:
                    params += list(self.scene_prior.parameters())
        else:
            params += list(self.to_z.parameters())
        params += list(self.decoder.parameters())
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _beta(self) -> float:
        if self.kl_warmup_steps <= 0:
            return self.kl_weight
        warm = min(1.0, float(self.global_step) / float(self.kl_warmup_steps))
        return self.kl_weight * warm

    def _slice_batch_agents(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sliced = dict(batch)
        if "agents_future" not in sliced:
            return sliced
        num_agents_in = int(sliced["agents_future"].shape[1])
        if num_agents_in < self.agents_len:
            raise ValueError(f"Batch has {num_agents_in} agents, expected >= {self.agents_len}")
        if num_agents_in == self.agents_len:
            return sliced
        for key in ["agents_history", "agents_future", "agents_interested", "agents_type", "agents_id", "anchors"]:
            if key in sliced and torch.is_tensor(sliced[key]) and sliced[key].dim() >= 2:
                sliced[key] = sliced[key][:, : self.agents_len]
        if "relations" in sliced and torch.is_tensor(sliced["relations"]) and sliced["relations"].dim() == 4:
            relations = sliced["relations"]
            n_total = int(relations.shape[1])
            keep = torch.cat(
                [
                    torch.arange(self.agents_len, device=relations.device),
                    torch.arange(num_agents_in, n_total, device=relations.device),
                ],
                dim=0,
            )
            sliced["relations"] = relations.index_select(1, keep).index_select(2, keep)
        return sliced

    def _prepare_gt(
        self,
        future_raw: torch.Tensor,
        agents_interested: torch.Tensor,
        agent_is_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_valid = (future_raw.abs().sum(dim=-1) > 0) & agent_is_valid.unsqueeze(-1)
        if self.truncate_after_invalid:
            # IMPORTANT: do not overwrite invalid future states with global zeros (0,0,0) because
            # after the query-frame transform those become huge artificial deltas that corrupt the encoder.
            state_valid = torch.cumprod(state_valid.to(torch.int32), dim=-1).to(torch.bool)

        query_idx = _pick_query_agent_index(agents_interested)
        traj_q = _transform_traj_to_query_frame(future_raw, query_idx=query_idx)
        traj_local = _transform_traj_to_agent_frame(future_raw)
        # Replace invalid timesteps with t0 pose so the encoder sees stable zeros after delta conversion.
        t0 = traj_q[:, :, :1, :].expand_as(traj_q)
        traj_q = torch.where(state_valid.unsqueeze(-1), traj_q, t0)
        t0_local = traj_local[:, :, :1, :].expand_as(traj_local)
        traj_local = torch.where(state_valid.unsqueeze(-1), traj_local, t0_local)
        traj_delta = _traj_to_deltas(traj_local)
        gt_sincos = _dtheta_to_sincos(traj_delta[..., 2])
        traj_delta_sincos = torch.cat([traj_delta[..., :2], gt_sincos], dim=-1)  # [B,A,T,4]
        return query_idx, traj_q, traj_local, traj_delta, traj_delta_sincos, state_valid

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        sample_latent: Optional[bool] = None,
        ablate_context: bool = False,
    ) -> BehaviorVAEOutputs:
        batch = self._slice_batch_agents(batch)
        if sample_latent is None:
            sample_latent = self.latent_sampling

        future_raw = batch["agents_future"][:, : self.agents_len, : self.future_len, :3].float()  # [B,A,T,3]
        if self.train_context_encoder:
            enc = self.context_encoder(batch)
        else:
            self.context_encoder.eval()
            with torch.no_grad():
                enc = self.context_encoder(batch)

        encodings = enc["encodings"]  # [B,N,256]
        agents_mask = enc["agents_mask"][:, : self.agents_len]
        agent_is_valid = (~agents_mask).to(encodings.device)
        context_agents = encodings[:, : self.agents_len]
        if not self.use_context:
            context_post = torch.zeros_like(context_agents)
            context_prior = torch.zeros_like(context_agents)
        elif ablate_context:
            context_post = torch.zeros_like(context_agents)
            context_prior = torch.zeros_like(context_agents)
        else:
            context_prior = context_agents
            if self.posterior_use_context:
                context_post = context_agents
            else:
                context_post = torch.zeros_like(context_agents)

        query_idx, traj_gt_q, traj_gt_local, traj_gt_delta, traj_gt_delta_sincos, state_valid = self._prepare_gt(
            future_raw, batch["agents_interested"][:, : self.agents_len], agent_is_valid=agent_is_valid
        )

        # Encode and couple across agents.
        feats = self.traj_enc(traj_gt_delta_sincos)  # [B,A,H]
        rel_geom = _relative_geometry(future_raw[:, :, 0], agent_is_valid=agent_is_valid)
        feats = self.interaction(feats, rel_geom=rel_geom, agent_is_valid=agent_is_valid)
        z_in = torch.cat([feats, context_post], dim=-1) if self.use_context else feats
        if self.use_vae:
            z_in = self.to_feat(z_in)
            mu = self.to_mu(z_in)
            log_var = self.to_logvar(z_in)
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            if sample_latent:
                z_agent = _reparameterize(mu, log_var)
            else:
                z_agent = mu

            mu_p = torch.zeros_like(mu)
            log_var_p = torch.zeros_like(log_var)
            if self.use_conditional_prior:
                c = self.prior_feat(context_prior)
                mu_p = self.prior_mu(c)
                log_var_p = torch.clamp(self.prior_logvar(c), min=-10.0, max=10.0)

            z_scene = None
            mu_scene = None
            log_var_scene = None
            mu_scene_p = None
            log_var_scene_p = None
            if self.use_z_scene:
                # Posterior scene latent from masked mean of agent features.
                valid_f = agent_is_valid.to(feats.dtype)
                denom = valid_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
                pooled = (z_in * valid_f.unsqueeze(-1)).sum(dim=1) / denom  # [B,in_z]
                params_s = self.scene_post(pooled)
                mu_scene, log_var_scene = params_s.chunk(2, dim=-1)
                log_var_scene = torch.clamp(log_var_scene, min=-10.0, max=10.0)
                if sample_latent:
                    z_scene = _reparameterize(mu_scene, log_var_scene)
                else:
                    z_scene = mu_scene

                if self.use_conditional_prior:
                    pooled_c = (context_prior * valid_f.unsqueeze(-1)).sum(dim=1) / denom  # [B,256]
                    params_sp = self.scene_prior(pooled_c)
                    mu_scene_p, log_var_scene_p = params_sp.chunk(2, dim=-1)
                    log_var_scene_p = torch.clamp(log_var_scene_p, min=-10.0, max=10.0)
        else:
            mu = torch.zeros_like(z_in[..., : self.z_dim])
            log_var = torch.zeros_like(mu)
            z_agent = self.to_z(z_in)
            mu_p = torch.zeros_like(mu)
            log_var_p = torch.zeros_like(log_var)
            z_scene = None
            mu_scene = None
            log_var_scene = None
            mu_scene_p = None
            log_var_scene_p = None

        traj_pred_delta = self.decoder(z_agent, z_scene)  # [B,A,T,4]

        return BehaviorVAEOutputs(
            query_idx=query_idx,
            agent_is_valid=agent_is_valid,
            state_valid=state_valid,
            traj_gt_q=traj_gt_q,
            traj_gt_local=traj_gt_local,
            traj_gt_delta=traj_gt_delta,
            z_agent=z_agent,
            mu=mu,
            log_var=log_var,
            mu_p=mu_p,
            log_var_p=log_var_p,
            z_scene=z_scene,
            mu_scene=mu_scene,
            log_var_scene=log_var_scene,
            mu_scene_p=mu_scene_p,
            log_var_scene_p=log_var_scene_p,
            traj_pred_delta=traj_pred_delta,
        )

    def _recon_loss(self, pred: torch.Tensor, gt: torch.Tensor, state_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pred/gt: [B,A,T,4] (dx,dy,sin,cos)
        valid = state_valid.to(torch.bool)
        denom = valid.float().sum().clamp(min=1.0)

        dxdy = F.mse_loss(pred[..., :2], gt[..., :2], reduction="none").sum(dim=-1)  # [B,A,T]
        sc = F.mse_loss(pred[..., 2:], gt[..., 2:], reduction="none").sum(dim=-1)  # [B,A,T]

        dxdy_m = (dxdy * valid.float()).sum() / denom
        sc_m = (sc * valid.float()).sum() / denom
        return dxdy_m + sc_m, dxdy_m, sc_m

    def _kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor, agent_is_valid: torch.Tensor) -> torch.Tensor:
        if self.use_conditional_prior:
            raise RuntimeError("Use _kl_agent_conditional in conditional-prior mode")
        kl_pa = _kl_diag_standard_normal(mu, log_var)  # [B,A]
        if self.free_nats > 0.0:
            kl_pa = torch.clamp(kl_pa, min=self.free_nats)
        denom = agent_is_valid.float().sum().clamp(min=1.0)
        return (kl_pa * agent_is_valid.float()).sum() / denom

    def _kl_agent_conditional(
        self,
        mu_q: torch.Tensor,
        log_var_q: torch.Tensor,
        mu_p: torch.Tensor,
        log_var_p: torch.Tensor,
        agent_is_valid: torch.Tensor,
    ) -> torch.Tensor:
        kl_pa = _kl_diag_gaussians(mu_q, log_var_q, mu_p, log_var_p)  # [B,A]
        if self.free_nats > 0.0:
            kl_pa = torch.clamp(kl_pa, min=self.free_nats)
        denom = agent_is_valid.float().sum().clamp(min=1.0)
        return (kl_pa * agent_is_valid.float()).sum() / denom

    def _kl_scene_conditional(
        self,
        mu_q: torch.Tensor,
        log_var_q: torch.Tensor,
        mu_p: torch.Tensor,
        log_var_p: torch.Tensor,
    ) -> torch.Tensor:
        kl = _kl_diag_gaussians(mu_q, log_var_q, mu_p, log_var_p)  # [B]
        if self.free_nats > 0.0:
            kl = torch.clamp(kl, min=self.free_nats)
        return kl.mean()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        sample_latent = self.latent_sampling and (self.global_step >= self.latent_sampling_warmup_steps)
        out = self.forward(batch, sample_latent=sample_latent)
        gt_sincos = _dtheta_to_sincos(out.traj_gt_delta[..., 2])
        gt = torch.cat([out.traj_gt_delta[..., :2], gt_sincos], dim=-1)
        recon, dxdy, sc = self._recon_loss(out.traj_pred_delta, gt, out.state_valid)

        kl = torch.tensor(0.0, device=self.device)
        beta = 0.0
        if self.use_vae and self.kl_weight > 0.0:
            if self.use_conditional_prior:
                kl = self._kl_agent_conditional(out.mu, out.log_var, out.mu_p, out.log_var_p, out.agent_is_valid)
                if self.use_z_scene:
                    kl = kl + self._kl_scene_conditional(out.mu_scene, out.log_var_scene, out.mu_scene_p, out.log_var_scene_p)
            else:
                kl = self._kl_loss(out.mu, out.log_var, out.agent_is_valid)
            beta = self._beta()
        loss = recon + beta * kl

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon", recon, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/dxdy", dxdy, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/sincos", sc, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/kl", kl, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/beta", torch.tensor(beta, device=self.device), prog_bar=False, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        val_latent_sampling = bool(self.cfg.get("val_latent_sampling", False))
        out = self.forward(batch, sample_latent=val_latent_sampling)
        gt_sincos = _dtheta_to_sincos(out.traj_gt_delta[..., 2])
        gt = torch.cat([out.traj_gt_delta[..., :2], gt_sincos], dim=-1)
        recon, dxdy, sc = self._recon_loss(out.traj_pred_delta, gt, out.state_valid)

        kl = torch.tensor(0.0, device=self.device)
        beta = 0.0
        if self.use_vae and self.kl_weight > 0.0:
            if self.use_conditional_prior:
                kl = self._kl_agent_conditional(out.mu, out.log_var, out.mu_p, out.log_var_p, out.agent_is_valid)
                if self.use_z_scene:
                    kl = kl + self._kl_scene_conditional(out.mu_scene, out.log_var_scene, out.mu_scene_p, out.log_var_scene_p)
            else:
                kl = self._kl_loss(out.mu, out.log_var, out.agent_is_valid)
            beta = self._beta()
        loss = recon + beta * kl

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon", recon, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/dxdy", dxdy, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/sincos", sc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/kl", kl, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss
