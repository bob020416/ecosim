import torch
import torch.nn as nn
import lightning.pytorch as pl
import math
from typing import List, Optional, Iterable, Tuple
from .modules import Encoder, Denoiser, GoalPredictor, Ctrl_Denoiser, Ctrl_TransformerDecoder
from .utils import DDPM_Sampler
from .model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import smooth_l1_loss, cross_entropy


# ============================================================================
# Text Prompt Encoder Components
# ============================================================================


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for LoRALinear")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return list(model.transformer.layer)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return list(model.encoder.layers)
    return []


def _iter_lora_targets(
    layer: nn.Module, target_suffixes: Tuple[str, ...]
) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(sfx) for sfx in target_suffixes):
            yield name, module


def _replace_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _apply_lora_to_model(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    layers: int,
    freeze_base: bool,
    target_suffixes: Optional[List[str]] = None,
) -> int:
    if rank <= 0:
        return 0
    targets = tuple(target_suffixes or ["q_lin", "k_lin", "v_lin", "out_lin", "q", "k", "v", "o"])
    tf_layers = _get_transformer_layers(model)
    if not tf_layers:
        return 0
    if layers and layers > 0:
        start = max(len(tf_layers) - layers, 0)
        layer_indices = range(start, len(tf_layers))
    else:
        layer_indices = range(len(tf_layers))

    replaced = 0
    for idx in layer_indices:
        layer = tf_layers[idx]
        for name, module in list(_iter_lora_targets(layer, targets)):
            _replace_module(
                layer,
                name,
                LoRALinear(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=freeze_base,
                ),
            )
            replaced += 1
    return replaced


class TextPromptEncoder(nn.Module):
    """Encode per-agent text prompts into conditioning features for FiLM."""

    def __init__(
        self,
        model_name_or_path: str,
        feature_dim: int,
        *,
        max_length: int = 128,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        use_mean_pool: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_layers: int = 0,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TextPromptEncoder. Install via `pip install transformers`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.feature_dim = int(feature_dim)
        self.max_length = int(max_length)
        self.use_mean_pool = bool(use_mean_pool)
        self.freeze_backbone = bool(freeze_backbone)

        hidden_size = int(self.model.config.hidden_size)
        if hidden_size != self.feature_dim:
            self.proj = nn.Linear(hidden_size, self.feature_dim)
        else:
            self.proj = nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            for p in self.model.pooler.parameters():
                p.requires_grad = False

        self._lora_applied = 0
        if lora_rank and lora_rank > 0:
            self._lora_applied = _apply_lora_to_model(
                self.model,
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                layers=int(lora_layers),
                freeze_base=self.freeze_backbone,
                target_suffixes=lora_target_modules,
            )

        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def _init_weights(self) -> None:
        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def _pool(self, last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_mean_pool and attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        return last_hidden[:, 0]

    def forward(
        self,
        prompts: List[str],
        *,
        device: Optional[torch.device] = None,
        **_: object,
    ) -> torch.Tensor:
        if prompts is None:
            raise ValueError("prompts must be a list of strings")

        n_total = len(prompts)
        if n_total == 0:
            dtype = next(self.model.parameters()).dtype
            return torch.zeros((0, self.feature_dim), device=device, dtype=dtype)

        if device is None:
            device = next(self.parameters()).device

        valid_idx = [i for i, p in enumerate(prompts) if isinstance(p, str) and p.strip()]
        out = torch.zeros(
            (n_total, self.feature_dim),
            device=device,
            dtype=next(self.model.parameters()).dtype,
        )
        if not valid_idx:
            return out

        texts = [prompts[i] for i in valid_idx]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        pooled = self._pool(outputs.last_hidden_state, encoded.get("attention_mask"))
        pooled = self.dropout(pooled)
        pooled = self.proj(pooled)
        idx = torch.tensor(valid_idx, device=device)
        out = out.to(pooled.dtype).index_copy(0, idx, pooled)
        return out


# ============================================================================
# Temporal Encoder Components
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal modeling."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [B, A, T, D]"""
        B, A, T, D = x.shape
        pe = self.pe[:T, :].unsqueeze(0).unsqueeze(0)
        return x + pe


class TemporalTransformerBlock(nn.Module):
    """Transformer block for temporal sequence modeling."""

    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """x: [B*A, T, D]"""
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class LearnedAttentionPooling(nn.Module):
    """Learned attention pooling over temporal dimension."""

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """x: [B, A, T, D] -> [B, A, D]"""
        B, A, T, D = x.shape
        x_flat = x.reshape(B*A, T, D)
        query = self.query.expand(B*A, 1, D)
        keys = self.key_proj(x_flat)
        values = self.value_proj(x_flat)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights, values).squeeze(1)
        return pooled.reshape(B, A, D)


class CrossAgentAttention(nn.Module):
    """
    Cross-agent attention for spatial modeling.

    Allows each agent's conditioning to "see" other agents' conditioning,
    enabling multi-agent coordination learning (e.g., yielding, following, overtaking).

    Architecture:
        Input [B, A, D]
          ↓ Self-Attention over A agents
        Output [B, A, D]
    """

    def __init__(self, d_model=64, n_heads=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, A, D] - per-agent features
        Returns:
            out: [B, A, D] - spatially-aware features
        """
        # Self-attention over agents dimension
        # Each agent attends to all other agents
        attn_out, _ = self.self_attn(x, x, x)
        out = self.norm(x + self.dropout(attn_out))
        return out


class FutureEncoder(nn.Module):
    """
    Temporal-aware encoder for future trajectory sketches.

    Uses lightweight transformers to preserve temporal structure and model
    trajectory dynamics (acceleration, turning, etc.). Designed for closed-loop
    training where temporal consistency matters.

    Architecture:
        Sketch [B, A, T, 3]
          ↓ Temporal Embedding
        [B, A, T, d_model]
          ↓ Positional Encoding
        [B, A, T, d_model]
          ↓ Temporal Transformer (n_layers)
        [B, A, T, d_model]
          ↓ Learned Attention Pooling
        [B, A, d_model]
          ↓ Cross-Agent Attention (optional)
        [B, A, d_model]

    Args:
        input_dim: Input feature dimension (default: 3 for x, y, heading)
        feature_dim: Output feature dimension (default: 64)
        agents_len: Number of agents
        dropout: Dropout rate for transformer
        n_layers: Number of transformer layers (default: 2)
        n_heads: Number of attention heads (default: 4)
        d_ff: Feed-forward dimension (default: 256)
        use_cross_agent_attn: Whether to use cross-agent attention (default: True)
    """

    def __init__(
        self,
        input_dim=3,
        feature_dim=64,
        agents_len=32,
        dropout: float = 0.0,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        use_cross_agent_attn: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.agents_len = agents_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_cross_agent_attn = use_cross_agent_attn

        # 1. Temporal embedding: project each timestep to d_model
        self.temporal_embed = nn.Linear(input_dim, feature_dim)

        # 2. Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(feature_dim, max_len=100)

        # 3. Temporal transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(
                d_model=feature_dim,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # 4. Learned attention pooling
        self.pool = LearnedAttentionPooling(feature_dim)

        # 5. Cross-agent attention (optional but recommended)
        if use_cross_agent_attn:
            self.cross_agent_attn = CrossAgentAttention(
                d_model=feature_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:
            self.cross_agent_attn = None

        # Input normalization
        self.input_norm = nn.LayerNorm(feature_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, agents_future):
        """
        Encode future trajectory sketches.

        Args:
            agents_future: [B, A, T, C] - future trajectories
                          C >= 3 (uses first 3: x, y, heading)

        Returns:
            features: [B, A, feature_dim] - encoded temporal features
        """
        B, A, T, C = agents_future.shape


        # Extract x, y, heading (first 3 channels)
        sketch = agents_future[..., :self.input_dim]  # [B, A, T, 3]

        # Convert to relative coordinates (relative to initial position)
        initial_pos = sketch[:, :, 0:1, :]  # [B, A, 1, 3]
        relative_sketch = sketch.clone()

        # Relative x, y
        relative_sketch[..., :2] = sketch[..., :2] - initial_pos[..., :2]

        # Relative heading (normalized to [-π, π])
        relative_sketch[..., 2] = sketch[..., 2] - initial_pos[..., 2]
        relative_sketch[..., 2] = torch.atan2(
            torch.sin(relative_sketch[..., 2]),
            torch.cos(relative_sketch[..., 2])
        )

        # 1. Temporal embedding
        x = self.temporal_embed(relative_sketch)  # [B, A, T, feature_dim]

        # 2. Add positional encoding
        x = self.pos_encoding(x)  # [B, A, T, feature_dim]

        # 3. Layer normalization
        x = self.input_norm(x)

        # 4. Reshape for transformer: process each agent independently
        x = x.reshape(B*A, T, self.feature_dim)  # [B*A, T, feature_dim]

        # 5. Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # [B*A, T, feature_dim]

        # 6. Reshape back
        x = x.reshape(B, A, T, self.feature_dim)  # [B, A, T, feature_dim]

        # 7. Learned attention pooling
        features = self.pool(x)  # [B, A, feature_dim]

        # 8. Cross-agent attention (spatial modeling)
        if self.cross_agent_attn is not None:
            features = self.cross_agent_attn(features)  # [B, A, feature_dim]


        return features


class LatentEncoder(nn.Module):
    """Project per-agent latent codes into ControlNet conditioning features.

    Design goals:
    - Keep the encoder simple and stable (MLP + optional cross-agent attention).
    - Allow using agent-only latents (recommended for behavior transfer), with optional scenario latents.
    - Output per-agent conditioning features [B, A, F] for FiLM injection.
    """

    def __init__(
        self,
        agent_latent_dim: int,
        feature_dim: int,
        *,
        scenario_latent_dim: int = 0,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        use_cross_agent_attn: bool = True,
        cross_agent_heads: int = 4,
    ):
        super().__init__()
        if agent_latent_dim <= 0:
            raise ValueError(f"agent_latent_dim must be >0, got {agent_latent_dim}")
        if scenario_latent_dim < 0:
            raise ValueError(f"scenario_latent_dim must be >=0, got {scenario_latent_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be >0, got {feature_dim}")
        if n_layers < 2:
            raise ValueError("n_layers must be >=2")
        if use_cross_agent_attn and (feature_dim % cross_agent_heads != 0):
            raise ValueError(f"feature_dim ({feature_dim}) must be divisible by cross_agent_heads ({cross_agent_heads})")

        self.feature_dim = int(feature_dim)
        self.agent_latent_dim = int(agent_latent_dim)
        self.scenario_latent_dim = int(scenario_latent_dim)
        self.input_dim = self.agent_latent_dim + self.scenario_latent_dim

        self.input_norm = nn.LayerNorm(self.input_dim)

        layers: list[nn.Module] = [nn.Linear(self.input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, self.feature_dim))
        self.mlp = nn.Sequential(*layers)

        # Optional cross-agent coupling on the conditioning tokens.
        self.cross_agent_attn = (
            CrossAgentAttention(d_model=self.feature_dim, n_heads=cross_agent_heads, dropout=dropout)
            if use_cross_agent_attn
            else None
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Make last layer small to start close to "no conditioning" (FiLM already starts at identity).
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight, gain=0.05)
            nn.init.zeros_(last.bias)

    def forward(self, agent_latents: torch.Tensor, scenario_latents: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            agent_latents: [B, A, Za] (required)
            scenario_latents: optional scene/context latents, either:
                - [B, A, Zs] (per-agent), or
                - [B, Zs] (per-scene, will be broadcast over agents)

        Returns:
            conditioning_features: [B, A, F]
        """
        if agent_latents.dim() != 3:
            raise ValueError(f"agent_latents must be [B,A,Za], got {agent_latents.shape}")
        b, a, za = agent_latents.shape
        if za != self.agent_latent_dim:
            raise ValueError(f"agent_latents last dim must be {self.agent_latent_dim}, got {za}")

        x = agent_latents
        if self.scenario_latent_dim > 0:
            if scenario_latents is None:
                raise ValueError("scenario_latents must be provided when scenario_latent_dim > 0")
            if scenario_latents.dim() == 2:
                if scenario_latents.shape != (b, self.scenario_latent_dim):
                    raise ValueError(f"scenario_latents must be [B,{self.scenario_latent_dim}], got {scenario_latents.shape}")
                scenario_latents = scenario_latents[:, None, :].expand(b, a, self.scenario_latent_dim)
            elif scenario_latents.dim() == 3:
                if scenario_latents.shape != (b, a, self.scenario_latent_dim):
                    raise ValueError(
                        f"scenario_latents must be [B,A,{self.scenario_latent_dim}], got {scenario_latents.shape}"
                    )
            else:
                raise ValueError(f"scenario_latents must be [B,Zs] or [B,A,Zs], got {scenario_latents.shape}")
            x = torch.cat([agent_latents, scenario_latents], dim=-1)

        x = self.input_norm(x)
        x = self.mlp(x)  # [B,A,F]
        if self.cross_agent_attn is not None:
            x = self.cross_agent_attn(x)
        return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning.

    Uses direct FiLM without scaling, initialized at identity (gamma=1, beta=0).
    This allows the model to learn appropriate modulation ranges freely.
    """

    def __init__(
        self,
        feature_dim,
        conditioning_dim,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim # feature is the query from denoiser
        self.conditioning_dim = conditioning_dim
        self._last_stats = {}

        # gamma_raw、beta_raw 就是「conditioning 經過 MLP 後」得到的新的兩個向量，各自對應 denoiser 特徵的每一個 channel
        mlp_layers = [
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            mlp_layers.append(nn.Dropout(dropout))
        mlp_layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])
        if dropout > 0.0:
            mlp_layers.append(nn.Dropout(dropout))
        mlp_layers.append(nn.Linear(hidden_dim, 2 * feature_dim))  # gamma and beta (raw)
        self.film_mlp = nn.Sequential(*mlp_layers)

        # Initialize to identity transformation
        self._init_weights()

    def _init_weights(self):
        # Initialize with small random values to ensure gradient flow
        with torch.no_grad():
            nn.init.xavier_uniform_(self.film_mlp[-1].weight, gain=0.01)
            self.film_mlp[-1].bias[:self.feature_dim].fill_(1.0)  # gamma starts around 1
            self.film_mlp[-1].bias[self.feature_dim:].fill_(0.0)  # beta starts around 0
            # Add small noise to bias to break perfect identity
            self.film_mlp[-1].bias.add_(torch.randn_like(self.film_mlp[-1].bias) * 0.001)

    def forward(self, x, conditioning):
        """
        Args:
            x: [B, A, T, F] - denoiser query input features
            conditioning: [B, A, C] - conditioning features
        Returns:
            modulated_x: [B, A, T, F] - FiLM modulated features
        """
        B, A, T, F = x.shape

        # Expand conditioning to match temporal dimension
        if conditioning.dim() == 4:
            conditioning_expanded = conditioning
            if conditioning_expanded.shape[2] != T:
                if conditioning_expanded.shape[2] > T:
                    conditioning_expanded = conditioning_expanded[:, :, -T:, :]
                else:
                    pad_len = T - conditioning_expanded.shape[2]
                    pad = torch.zeros(
                        B,
                        A,
                        pad_len,
                        conditioning_expanded.shape[3],
                        device=conditioning.device,
                        dtype=conditioning.dtype,
                    )
                    conditioning_expanded = torch.cat([pad, conditioning_expanded], dim=2)
        else:
            conditioning_expanded = conditioning.unsqueeze(2).expand(B, A, T, -1)  # [B, A, T, C]
        conditioning_flat = conditioning_expanded.reshape(B*A*T, -1)

        # Generate gamma and beta
        film_params = self.film_mlp(conditioning_flat)  # [B*A*T, 2*F]
        gamma, beta = film_params.split(F, dim=1)  # Each [B*A*T, F]

        # Reshape back
        gamma = gamma.reshape(B, A, T, F)
        beta = beta.reshape(B, A, T, F)

        # No scaling - let model learn appropriate range freely
        # Initialized at gamma=1, beta=0 for identity transformation

        with torch.no_grad():
            gamma_detached = gamma.detach()
            beta_detached = beta.detach()
            self._last_stats = {
                'gamma_mean': gamma_detached.mean().item(),
                'gamma_std': gamma_detached.std(unbiased=False).item(),
                'gamma_min': gamma_detached.min().item(),
                'gamma_max': gamma_detached.max().item(),
                'beta_mean': beta_detached.mean().item(),
                'beta_std': beta_detached.std(unbiased=False).item(),
                'beta_min': beta_detached.min().item(),
                'beta_max': beta_detached.max().item(),
                'gamma_residual_mean': (gamma_detached - 1.0).abs().mean().item(),
                'beta_abs_mean': beta_detached.abs().mean().item(),
            }

        # Apply FiLM: γ(c) ⊙ x + β(c)
        modulated_x = gamma * x + beta

        return modulated_x

    @property
    def last_stats(self):
        return self._last_stats


class ControlNetModule(nn.Module):
    """
    ControlNet module that provides FiLM conditioning to the original denoiser.
    This module is trainable while the original model is frozen.
    Supports two conditioning modes: 'future' (trajectory) or 'latent' (agent latent conditioning).

    New in this version:
    - FutureEncoder uses lightweight temporal transformers (default feature_dim=64)
    - Preserves temporal structure of trajectory sketches
    - Reduced parameters: ~1.2M (down from 3.17M)
    """
    def __init__(
        self,
        future_feature_dim=64,
        agents_len=32,
        film_blocks=[0, 1],
        future_dropout: float = 0.0,
        film_dropout: float = 0.0,
        conditioning_mode='future',  # 'future' or 'latent'
        # Temporal encoder parameters
        temporal_n_layers: int = 2,
        temporal_n_heads: int = 4,
        temporal_d_ff: int = 256,
        use_cross_agent_attn: bool = True,  # Enable cross-agent spatial modeling
        latent_use_cross_agent_attn: bool | None = None,
        # Latent encoder parameters
        agent_latent_dim: int = 16,
        scenario_latent_dim: int = 0,
        latent_hidden_dim: int = 256,
        latent_n_layers: int = 3,
        latent_cross_agent_heads: int = 4,
        # Text encoder parameters (conditioning_mode='text')
        text_model_path: str = "ckpt",
        text_max_length: int = 128,
        text_freeze_backbone: bool = True,
        text_use_mean_pool: bool = True,
        text_encoder_dropout: float = 0.0,
        text_lora_rank: int = 0,
        text_lora_alpha: float = 16.0,
        text_lora_dropout: float = 0.0,
        text_lora_layers: int = 0,
        text_lora_target_modules: Optional[List[str]] = None,
        # Optional time-aware conditioning
        condition_time_embed: bool = False,
        time_embed_max_len: int = 100,
    ):
        super().__init__()
        self._future_feature_dim = future_feature_dim
        self._agents_len = agents_len
        self._film_blocks = film_blocks
        self._conditioning_mode = conditioning_mode
        self._condition_time_embed = bool(condition_time_embed)
        self._time_embed = (
            SinusoidalPositionalEncoding(future_feature_dim, max_len=time_embed_max_len)
            if self._condition_time_embed
            else None
        )
        self._future_conditioning_mode = "rolling"

        # Conditioning encoder (choose one based on mode)
        if conditioning_mode == 'future':
            # Temporal-aware FutureEncoder with transformers + cross-agent attention
            self.future_encoder = FutureEncoder(
                input_dim=3,
                feature_dim=future_feature_dim,
                agents_len=agents_len,
                dropout=future_dropout,
                n_layers=temporal_n_layers,
                n_heads=temporal_n_heads,
                d_ff=temporal_d_ff,
                use_cross_agent_attn=use_cross_agent_attn,
            )
            self.latent_encoder = None
            self.text_encoder = None
        elif conditioning_mode == 'latent':
            if latent_use_cross_agent_attn is None:
                latent_use_cross_agent_attn = use_cross_agent_attn
            # LatentEncoder for per-agent latent conditioning
            self.latent_encoder = LatentEncoder(
                agent_latent_dim=agent_latent_dim,
                scenario_latent_dim=scenario_latent_dim,
                feature_dim=future_feature_dim,
                hidden_dim=latent_hidden_dim,
                n_layers=latent_n_layers,
                dropout=future_dropout,
                use_cross_agent_attn=latent_use_cross_agent_attn,
                cross_agent_heads=latent_cross_agent_heads,
            )
            self.future_encoder = None
            self.text_encoder = None
        elif conditioning_mode == 'text':
            self.text_encoder = TextPromptEncoder(
                model_name_or_path=text_model_path,
                feature_dim=future_feature_dim,
                max_length=text_max_length,
                dropout=float(text_encoder_dropout),
                freeze_backbone=text_freeze_backbone,
                use_mean_pool=text_use_mean_pool,
                lora_rank=text_lora_rank,
                lora_alpha=text_lora_alpha,
                lora_dropout=text_lora_dropout,
                lora_layers=text_lora_layers,
                lora_target_modules=text_lora_target_modules,
            )
            self.future_encoder = None
            self.latent_encoder = None
        else:
            raise ValueError(
                f"Unknown conditioning_mode: {conditioning_mode}. Must be 'future', 'latent', or 'text'."
            )

        # FiLM layers for different blocks
        # TransformerDecoder uses 256 as feature dim
        # Conditioning dim is future_feature_dim (64 by default)
        self.film_layers = nn.ModuleDict()
        for block_idx in film_blocks:
            self.film_layers[f'block_{block_idx}'] = FiLMLayer(
                feature_dim=256,  # TransformerDecoder feature dimension
                conditioning_dim=future_feature_dim,  # 64 instead of 256
                hidden_dim=512,
                dropout=film_dropout,
            )
        self._last_stats = {}

    def encode_future(self, agents_future):
        """
        DEPRECATED: Use encode_conditioning() instead.
        Kept for backward compatibility.
        """
        return self.encode_conditioning(agents_future=agents_future)

    def encode_conditioning(
        self,
        agents_future=None,
        agent_latents=None,
        scenario_latents=None,
        text_prompt=None,
        text_prompt_mask=None,
    ):
        """
        Encode conditioning features based on the conditioning mode.

        Args:
            agents_future: [B, A, T, 3] - future agent trajectories (for 'future' mode)
            agent_latents: [B, A, Za] - per-agent latent embeddings (for 'latent' mode)
            scenario_latents: optional latents for context (for 'latent' mode), [B,Zs] or [B,A,Zs]

        Returns:
            conditioning_features: [B, A, F] - encoded conditioning features
        """
        if self._conditioning_mode == 'future':
            if agents_future is None:
                raise ValueError("agents_future is required when conditioning_mode='future'")

            return self.future_encoder(agents_future)

        elif self._conditioning_mode == 'latent':
            if agent_latents is None:
                raise ValueError("agent_latents is required when conditioning_mode='latent'")
            return self.latent_encoder(agent_latents, scenario_latents)

        elif self._conditioning_mode == 'text':
            if text_prompt is None:
                raise ValueError("text_prompt is required when conditioning_mode='text'")
            return self.encode_text_conditioning(text_prompt, text_prompt_mask)

        else:
            raise ValueError(f"Unknown conditioning_mode: {self._conditioning_mode}")

    def _flatten_text_prompts(self, text_prompt):
        if text_prompt is None:
            return [], 0, 0
        if isinstance(text_prompt, (list, tuple)) and text_prompt and isinstance(text_prompt[0], str):
            return list(text_prompt), 1, len(text_prompt)
        if not isinstance(text_prompt, (list, tuple)):
            raise ValueError("text_prompt must be a list of strings or list of list of strings")
        batch_size = len(text_prompt)
        if batch_size == 0:
            return [], 0, 0
        max_len = max(len(row) for row in text_prompt)
        flat = []
        for row in text_prompt:
            row_list = list(row)
            if len(row_list) < max_len:
                row_list.extend([""] * (max_len - len(row_list)))
            flat.extend(row_list)
        return flat, batch_size, max_len

    def encode_text_conditioning(self, text_prompt, text_prompt_mask=None):
        if self.text_encoder is None:
            raise ValueError("text_encoder is not initialized")
        # Normalize prompts to match model agents_len when possible.
        if isinstance(text_prompt, (list, tuple)) and text_prompt and not isinstance(text_prompt[0], str):
            target_len = self._agents_len
            normalized = []
            for row in text_prompt:
                row_list = list(row)
                if target_len and target_len > 0:
                    if len(row_list) < target_len:
                        row_list.extend([""] * (target_len - len(row_list)))
                    elif len(row_list) > target_len:
                        row_list = row_list[:target_len]
                normalized.append(row_list)
            text_prompt = normalized

        flat, batch_size, agents_len = self._flatten_text_prompts(text_prompt)
        if batch_size == 0 or agents_len == 0:
            dtype = next(self.text_encoder.parameters()).dtype
            device = next(self.text_encoder.parameters()).device
            return torch.zeros((batch_size, agents_len, self._future_feature_dim), device=device, dtype=dtype)
        feat = self.text_encoder(flat)
        feat = feat.reshape(batch_size, agents_len, -1)
        if text_prompt_mask is not None:
            mask = text_prompt_mask
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            # Align mask to prompts if needed
            if mask.shape[1] != agents_len:
                if mask.shape[1] > agents_len:
                    mask = mask[:, :agents_len]
                else:
                    pad_len = agents_len - mask.shape[1]
                    pad = torch.zeros(
                        mask.shape[0],
                        pad_len,
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    mask = torch.cat([mask, pad], dim=1)
            if mask.shape[0] != batch_size:
                if batch_size % mask.shape[0] == 0:
                    repeat = batch_size // mask.shape[0]
                    mask = mask.repeat_interleave(repeat, dim=0)
                else:
                    raise ValueError(
                        f"text_prompt_mask shape {mask.shape} does not match prompts (B={batch_size}, A={agents_len})"
                    )
            mask = mask.to(feat.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            feat = feat * mask.unsqueeze(-1)
        return feat

    def maybe_add_time_embed(self, conditioning, replan_idx: Optional[int] = None):
        if conditioning is None:
            return conditioning
        if not self._condition_time_embed or self._time_embed is None:
            return conditioning
        step = int(replan_idx) if replan_idx is not None else 0
        max_idx = int(self._time_embed.pe.shape[0] - 1)
        step = max(0, min(step, max_idx))
        emb = self._time_embed.pe[step].to(device=conditioning.device, dtype=conditioning.dtype)
        if conditioning.dim() == 4:
            return conditioning + emb.view(1, 1, 1, -1)
        if conditioning.dim() == 3:
            return conditioning + emb.view(1, 1, -1)
        if conditioning.dim() == 2:
            return conditioning + emb.view(1, -1)
        raise ValueError(f"conditioning must be 2D/3D/4D, got {conditioning.shape}")

    def apply_film_conditioning(self, features, conditioning_features, block_idx):
        """
        Apply FiLM conditioning to features at a specific block.

        Args:
            features: [B, A, T, 256] - features to be modulated
            conditioning_features: [B, A, F] - conditioning features
            block_idx: int - which block this is

        Returns:
            modulated_features: [B, A, T, 256] - FiLM modulated features
        """
        if f'block_{block_idx}' in self.film_layers:
            return self.film_layers[f'block_{block_idx}'](features, conditioning_features)
        else:
            return features  # No conditioning for this block

    def get_film_layers(self):
        return {name: layer for name, layer in self.film_layers.items()}

    def get_film_stats(self):
        stats = {}
        for name, layer in self.film_layers.items():
            layer_stats = getattr(layer, 'last_stats', None)
            if layer_stats:
                stats[name] = layer_stats
        return stats
 


class VBD(pl.LightningModule):
    """
    Versertile Behavior Diffusion model.
    """

    def __init__(
        self,
        cfg: dict,
    ):
        """
        Initialize the VBD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self._future_len = cfg['future_len']
        self._agents_len = cfg['agents_len']
        self._action_len = cfg['action_len']
        self._diffusion_steps = cfg['diffusion_steps']
        self._encoder_layers = cfg['encoder_layers']
        self._encoder_version = cfg.get('encoder_version', 'v1')
        self._action_mean = cfg['action_mean']
        self._action_std = cfg['action_std']
        self._clamp_value = cfg.get('clamp_value', 5.0)
        
        
        self._train_encoder = cfg.get('train_encoder', True)
        self._train_agent_encoder_only = cfg.get('train_agent_encoder_only', False)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', True)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._replay_buffer = cfg.get('replay_buffer', False)
        self._embeding_dim = cfg.get('embeding_dim', 5) # By default, the embed is the noised trajectory so the dimension is 5

        # ControlNet specific parameters
        self._use_controlnet = cfg.get('use_controlnet', True)
        self._future_feature_dim = cfg.get('future_feature_dim', 256)
        self._film_blocks = cfg.get('film_blocks', [0, 1])  # Which denoiser blocks to apply FiLM to
        self._freeze_original = cfg.get('freeze_original', True)  # Freeze original model weights
        self._future_dropout = cfg.get('controlnet_future_dropout', 0.0)
        self._film_dropout = cfg.get('controlnet_film_dropout', 0.0)
        self._control_time_dim = cfg.get('controlnet_time_dim', 0)
        # NEW: Conditioning mode selection
        self._conditioning_mode = cfg.get('conditioning_mode', 'future')  # 'future', 'latent', or 'text'
        # Latent conditioning dims (only used when conditioning_mode='latent')
        self._control_agent_latent_dim = int(cfg.get('control_agent_latent_dim', 16))
        # Set 0 for agent-only latent conditioning (recommended for behavior transfer).
        self._control_scenario_latent_dim = int(cfg.get('control_scenario_latent_dim', 0))
        # Closed-loop latent indexing:
        # - "window": agent_latents_segments[k] corresponds to the k-th planning window (recommended).
        # - "time": seg_idx = current_time // latent_segment_len (legacy).
        self._latent_index_mode = str(cfg.get('latent_index_mode', 'window')).lower()
        self._latent_segment_len = int(cfg.get('latent_segment_len', 27))
        # ControlNet conditioning configuration
        # Randomly drop a portion of agent futures during training to simulate missing conditioning
        self._per_agent_dropout_min = cfg.get('per_agent_dropout_min', 0.0)
        self._per_agent_dropout_max = cfg.get('per_agent_dropout_max', 0.0)
        # Optional interval (in training steps) to log guided/unguided metrics without heavy overhead every step.
        self._control_metric_interval = 0  # disable train-time guided/unguided logging by default
        # Text conditioning parameters
        self._control_text_model_path = str(cfg.get('text_model_path', 'ckpt'))
        self._control_text_max_length = int(cfg.get('text_max_length', 128))
        self._control_text_freeze_backbone = bool(cfg.get('text_freeze_backbone', True))
        self._control_text_use_mean_pool = bool(cfg.get('text_use_mean_pool', True))
        self._control_text_encoder_dropout = float(cfg.get('text_encoder_dropout', 0.0))
        self._control_text_lora_rank = int(cfg.get('text_lora_rank', 0))
        self._control_text_lora_alpha = float(cfg.get('text_lora_alpha', 16.0))
        self._control_text_lora_dropout = float(cfg.get('text_lora_dropout', 0.0))
        self._control_text_lora_layers = int(cfg.get('text_lora_layers', 0))
        self._control_text_lora_targets = cfg.get('text_lora_target_modules', None)
        # Optional time-aware embedding for static conditioning (e.g., text)
        self._control_condition_time_embed = bool(cfg.get('controlnet_condition_time_embed', False))
        self._control_time_embed_max_len = int(cfg.get('controlnet_time_embed_max_len', 100))
        self._future_conditioning_mode = str(
            cfg.get('future_conditioning_mode', 'rolling')
        ).lower()
        if self._future_conditioning_mode not in {'rolling', 'static_full'}:
            raise ValueError(
                "future_conditioning_mode must be 'rolling' or 'static_full', "
                f"got {self._future_conditioning_mode}"
            )

        # Closed-loop training parameters
        self._use_closed_loop_training = cfg.get('use_closed_loop_training', False)
        self._closed_loop_execution_horizon = cfg.get('closed_loop_execution_horizon', 11)  # Execution step (e.g. 11 steps = 1.1s)
        self._closed_loop_prediction_horizon = cfg.get('closed_loop_prediction_horizon', 40) # Prediction window (e.g. 40 steps = 4.0s)
        self._closed_loop_num_candidates = cfg.get('closed_loop_num_candidates', 5)  # M
        self._closed_loop_noise_scale = cfg.get('closed_loop_noise_scale', 1.0)  # γ
        self._dt = cfg.get('dt', 0.1)
        # Collision loss is currently disabled.
        self._collision_loss_weight = 0.0

        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)

        # Use ControlNet-enabled denoiser
        from .modules import Ctrl_Denoiser
        self.denoiser = Ctrl_Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
            input_dim=self._embeding_dim,
        )

        # ControlNet module (trainable)
        if self._use_controlnet:
            latent_use_cross_agent_attn = cfg.get('latent_use_cross_agent_attn', None)
            if latent_use_cross_agent_attn is None:
                latent_use_cross_agent_attn = cfg.get('use_cross_agent_attn', True)
            self.controlnet = ControlNetModule(
                future_feature_dim=self._future_feature_dim,
                agents_len=self._agents_len,
                film_blocks=self._film_blocks,
                future_dropout=self._future_dropout,
                film_dropout=self._film_dropout,
                conditioning_mode=self._conditioning_mode,
                # Temporal encoder parameters
                temporal_n_layers=cfg.get('temporal_n_layers', 2),
                temporal_n_heads=cfg.get('temporal_n_heads', 4),
                temporal_d_ff=cfg.get('temporal_d_ff', 256),
                use_cross_agent_attn=cfg.get('use_cross_agent_attn', True),
                latent_use_cross_agent_attn=latent_use_cross_agent_attn,
                # Latent encoder parameters
                agent_latent_dim=self._control_agent_latent_dim,
                scenario_latent_dim=self._control_scenario_latent_dim,
                latent_hidden_dim=int(cfg.get('control_latent_hidden_dim', 256)),
                latent_n_layers=int(cfg.get('control_latent_n_layers', 3)),
                latent_cross_agent_heads=int(cfg.get('control_latent_cross_agent_heads', 4)),
                # Text encoder parameters
                text_model_path=self._control_text_model_path,
                text_max_length=self._control_text_max_length,
                text_freeze_backbone=self._control_text_freeze_backbone,
                text_use_mean_pool=self._control_text_use_mean_pool,
                text_encoder_dropout=self._control_text_encoder_dropout,
                text_lora_rank=self._control_text_lora_rank,
                text_lora_alpha=self._control_text_lora_alpha,
                text_lora_dropout=self._control_text_lora_dropout,
                text_lora_layers=self._control_text_lora_layers,
                text_lora_target_modules=self._control_text_lora_targets,
                # Optional time-aware embedding
                condition_time_embed=self._control_condition_time_embed,
                time_embed_max_len=self._control_time_embed_max_len,
            )
            self.controlnet._future_conditioning_mode = self._future_conditioning_mode
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s = cfg.get('schedule_s', 0.0),
            e = cfg.get('schedule_e', 1.0),
            tau = cfg.get('schedule_tau', 1.0),
            scale = cfg.get('schedule_scale', 1.0),
        )
                
        self.register_buffer('action_mean', torch.tensor(self._action_mean))  
        self.register_buffer('action_std', torch.tensor(self._action_std))
    
    ################### Training Setup ###################
    def configure_optimizers(self):
        '''
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
        # ControlNet training strategy: freeze original model, train only ControlNet
        if self._use_controlnet and self._freeze_original:
            # Freeze all original model parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.denoiser.parameters():
                param.requires_grad = False
            if self._with_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

            # Only ControlNet module is trainable
            params_to_update = list(self.controlnet.parameters())
            print(f"ControlNet mode: Training only {len(params_to_update)} ControlNet parameters")

        else:
            # Original training strategy
            if not self._train_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                # If train_agent_encoder_only is True, unfreeze only agent_encoder
                if self._train_agent_encoder_only:
                    for param in self.encoder.agent_encoder.parameters():
                        param.requires_grad = True
                    print(f"Training only agent_encoder (keeping rest of encoder frozen)")
            if not self._train_denoiser:
                for param in self.denoiser.parameters():
                    param.requires_grad = False
            if self._with_predictor and (not self._train_predictor):
                for param in self.predictor.parameters():
                    param.requires_grad = False

            params_to_update = []
            for param in self.parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        # Use extremely small learning rate for ControlNet to prevent weight corruption
        if self._use_controlnet and self._freeze_original:
            controlnet_lr = self.cfg.get('controlnet_lr', 1e-4)
        else:
            controlnet_lr = self.cfg['lr']
        print(f"Using learning rate: {controlnet_lr} (ControlNet mode: {self._use_controlnet and self._freeze_original})")

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=controlnet_lr,
            weight_decay=self.cfg['weight_decay']
        )
        
        lr_warmpup_step = self.cfg['lr_warmup_step']
        lr_step_freq = self.cfg['lr_step_freq']
        lr_step_gamma = self.cfg['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
        
            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1
        
            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, 
                lr_warmpup_step, 
                lr_step_freq,
                lr_step_gamma,
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the VBD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary containing the model outputs.
        """
        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)
        
        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(encoder_outputs, noised_actions_normalized, diffusion_step)
            output_dict.update(denoiser_outputs)
            
        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)
            
        return output_dict
        
    def forward_denoiser(
        self,
        encoder_outputs,
        noised_actions_normalized,
        diffusion_step,
        agents_future=None,
        agent_latents=None,       # NEW: for latent conditioning
        scenario_latents=None,    # optional (deprecated; keep None for agent-only control)
        text_prompt=None,
        text_prompt_mask=None,
        replan_idx: Optional[int] = None,
        control_mask=None,
        add_noise=True,           # NEW: control noise in rollout
    ):
        """
        Forward pass of the denoiser module with optional ControlNet conditioning.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.
            agents_future: [B, A, T, 3] - future trajectories for ControlNet conditioning (optional, for 'future' mode).
            agent_latents: [B, A, Za] - agent latent embeddings (optional, for 'latent' mode).
            scenario_latents: optional latents for context (optional, for 'latent' mode), [B,Zs] or [B,A,Zs].
            text_prompt: list[str] or list[list[str]] - per-agent text prompts (optional, for 'text' mode).
            text_prompt_mask: [B, A] - mask for valid text prompts (optional).
            control_mask: [B, A] - mask for conditioning (optional).
            add_noise: Whether to add noise in rollout. Set to False for closed-loop training.

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """
        noised_actions = self.unnormalize_actions(noised_actions_normalized)

        # Get ControlNet conditioning if available
        future_conditioning = None
        film_layers = None
        if self._use_controlnet:
            # Use the new encode_conditioning method that handles both modes
            future_conditioning = self.controlnet.encode_conditioning(
                agents_future=agents_future,
                agent_latents=agent_latents,
                scenario_latents=scenario_latents,
                text_prompt=text_prompt,
                text_prompt_mask=text_prompt_mask,
            )
            if (
                self._conditioning_mode == 'text'
                or (
                    self._conditioning_mode == 'future'
                    and self._future_conditioning_mode == 'static_full'
                )
            ):
                future_conditioning = self.controlnet.maybe_add_time_embed(
                    future_conditioning, replan_idx=replan_idx
                )
            film_layers = self.controlnet.get_film_layers()

        if future_conditioning is not None and control_mask is not None:
            # Align control_mask shape to conditioning [B,A,*]
            if control_mask.dim() == 1:
                control_mask = control_mask.unsqueeze(0)
            if control_mask.shape[1] != future_conditioning.shape[1]:
                if control_mask.shape[1] > future_conditioning.shape[1]:
                    control_mask = control_mask[:, : future_conditioning.shape[1]]
                else:
                    pad_len = future_conditioning.shape[1] - control_mask.shape[1]
                    pad = torch.zeros(
                        control_mask.shape[0],
                        pad_len,
                        device=control_mask.device,
                        dtype=control_mask.dtype,
                    )
                    control_mask = torch.cat([control_mask, pad], dim=1)
            if control_mask.shape[0] != future_conditioning.shape[0]:
                if future_conditioning.shape[0] % control_mask.shape[0] == 0:
                    repeat = future_conditioning.shape[0] // control_mask.shape[0]
                    control_mask = control_mask.repeat_interleave(repeat, dim=0)
                else:
                    raise ValueError(
                        f"control_mask batch {control_mask.shape[0]} does not match conditioning batch {future_conditioning.shape[0]}"
                    )

        # Forward through ControlNet-enabled denoiser
        denoiser_output = self.denoiser(
            encoder_outputs, noised_actions, diffusion_step,
            future_conditioning=future_conditioning,
            film_layers=film_layers,
            control_mask=control_mask,
        )

        denoised_actions_normalized = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_step,
            noised_actions_normalized,
            prediction_type=self._prediction_type
        )
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out (with optional noise control for closed-loop training)
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_trajs = roll_out(
            current_states, denoised_actions,
            action_len=self.denoiser._action_len,
            global_frame=True,
            add_noise=add_noise
        )

        return {
            'denoiser_output': denoiser_output,
            'denoised_actions_normalized': denoised_actions_normalized,
            'denoised_actions': denoised_actions,
            'denoised_trajs': denoised_trajs,
        }
    
    def forward_predictor(self, encoder_outputs):
        """
        Forward pass of the predictor module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Predict goal
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        
        current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        goal_actions = self.unnormalize_actions(goal_actions_normalized)    
        goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
                    action_len=self.predictor._action_len, global_frame=True)
        
        return {
            'goal_actions_normalized': goal_actions_normalized,
            'goal_actions': goal_actions,
            'goal_scores': goal_scores,
            'goal_trajs': goal_trajs,
        }
        
    def forward_and_get_loss(self, batch, prefix = '', debug = False):
        """
        Forward pass of the model and compute the loss.

        Args:
            batch: Input batch.
            prefix: Prefix for the loss keys.
            debug: Flag to enable debug mode.

        Returns:
            total_loss: Total loss.
            log_dict: Dictionary containing the loss values.
            debug_outputs: Dictionary containing debug outputs.
        """
        # data inputs
        agents_future = batch['agents_future'][:, :self._agents_len]
        # Open-loop training must match the model horizon. The dataset can carry
        # longer futures (e.g. 81 steps) while the model is configured for a
        # shorter horizon (e.g. 41). If we do not crop here, inverse kinematics
        # produces more actions than the decoder expects and the decoder reshape
        # silently turns the extra timesteps into a fake batch dimension.
        if agents_future.shape[2] > self._future_len:
            agents_future = agents_future[:, :, :self._future_len]

        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch['agents_interested'][:, :self._agents_len]
        anchors = batch['anchors'][:, :self._agents_len]

        # Load agent latents if in latent mode
        agent_latents = None
        if self._conditioning_mode == 'latent':
            agent_latents = batch.get('agent_latents', None)
            if agent_latents is not None:
                agent_latents = agent_latents[:, :self._agents_len]
        text_prompt = None
        text_prompt_mask = None
        if self._conditioning_mode == 'text':
            text_prompt = batch.get('text_prompt', None)
            text_prompt_mask = batch.get('text_prompt_mask', None)

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        control_agents_future = agents_future
        control_agent_latents = agent_latents
        control_scenario_latents = None
        control_dropout_mask = None
        # Base mask: agents that both interest us and have valid future trajectories
        valid_agents = (agents_interested > 0)
        valid_agents = valid_agents & agents_future_valid.any(dim=-1)
        control_mask = valid_agents.float()
        if self._conditioning_mode == 'text' and text_prompt_mask is not None:
            text_mask = text_prompt_mask
            if text_mask.dim() == 3 and text_mask.shape[0] == 1:
                text_mask = text_mask.squeeze(0)
            if text_mask.dim() == 1:
                text_mask = text_mask.unsqueeze(0)
            if text_mask.shape == control_mask.shape:
                if text_mask.dtype != torch.bool:
                    text_mask = text_mask > 0
                control_mask = control_mask * text_mask.float()
        if (
            self.training
            and self._use_controlnet
            and prefix.startswith('train/')
            and self._per_agent_dropout_max > 0.0
        ):
            # Sample a dropout probability for this batch within the configured range
            dropout_prob = torch.empty(1, device=agents_future.device).uniform_(
                self._per_agent_dropout_min, self._per_agent_dropout_max
            ).item()
            dropout_prob = float(max(0.0, min(1.0, dropout_prob)))

            if dropout_prob > 0.0:
                if valid_agents.any():
                    random_tensor = torch.rand_like(valid_agents, dtype=torch.float32)
                    # Drop per-agent conditioning by zeroing their conditioning data
                    control_dropout_mask = (random_tensor < dropout_prob) & valid_agents

                    for b in range(control_dropout_mask.shape[0]):
                        valid_mask_b = valid_agents[b]
                        if valid_mask_b.any():
                            dropped_b = control_dropout_mask[b] & valid_mask_b
                            if dropped_b.all():
                                # Ensure at least one conditioned agent remains per batch
                                keep_idx = torch.nonzero(valid_mask_b, as_tuple=False)[0]
                                control_dropout_mask[b, keep_idx] = False

                    # Apply dropout based on conditioning mode
                    if self._conditioning_mode == 'future':
                        # Zero out future trajectories for dropped agents
                        control_agents_future = agents_future.clone()
                        dropout_expanded = control_dropout_mask.unsqueeze(-1).unsqueeze(-1)
                        dropout_expanded = dropout_expanded.expand_as(control_agents_future)
                        control_agents_future = control_agents_future.masked_fill(dropout_expanded, 0.0)

                    elif self._conditioning_mode == 'latent' and agent_latents is not None:
                        # Zero out latents for dropped agents
                        control_agent_latents = agent_latents.clone()
                        dropout_expanded_agent = control_dropout_mask.unsqueeze(-1).expand_as(control_agent_latents)
                        control_agent_latents = control_agent_latents.masked_fill(dropout_expanded_agent, 0.0)

                    dropped = (control_dropout_mask & valid_agents).sum().item()
                    total_valid = valid_agents.sum().item()
                    dropout_ratio = dropped / total_valid if total_valid > 0 else 0.0

                    log_dict[f'{prefix}control_dropout_ratio'] = dropout_ratio

                    # Expose mask for debugging / visualization
                    debug_outputs['control_dropout_mask'] = control_dropout_mask.detach().clone()

                    # Mask marks agents with available conditioning (1) vs dropped (0)
                    control_mask = (~control_dropout_mask & valid_agents).float()
                else:
                    control_mask.zero_()
            # If dropout prob resolves to 0.0, keep the default control_mask (valid_agents)

        # get actions from trajectory
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len,
        )
        
        gt_actions_normalized = self.normalize_actions(gt_actions)
        B, A, T, D = gt_actions_normalized.shape

        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        
        ############### Denoise #################
        if self._train_denoiser:
            
            diffusion_steps = torch.randint(
                0, self.noise_scheduler.num_steps, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, A).view(B, A, 1, 1)
            
            # sample noise 
            # noise = torch.randn(B*A, T, D).type_as(agents_future)
            noise = torch.randn(B, A, T, D).type_as(agents_future)
            
            # noise the input
            noised_action_normalized = self.noise_scheduler.add_noise(
                gt_actions_normalized, #.reshape(B*A, T, D),
                noise,
                diffusion_steps#, .reshape(B*A),
            )#.reshape(B, A, T, D)
            # noise = noise.reshape(B, A, T, D)

            if self._replay_buffer:
                with torch.no_grad():
                    # Forward for one step
                    denoise_outputs = self.forward_denoiser(
                        encoder_outputs, gt_actions_normalized, diffusion_steps.view(B,A),
                        agents_future=control_agents_future if self._use_controlnet else None,
                        agent_latents=control_agent_latents if self._use_controlnet else None,
                        scenario_latents=control_scenario_latents if self._use_controlnet else None,
                        text_prompt=text_prompt if self._use_controlnet else None,
                        text_prompt_mask=text_prompt_mask if self._use_controlnet else None,
                        replan_idx=0,
                        control_mask=control_mask if self._use_controlnet else None,
                    )
                    
                    x_0 = denoise_outputs['denoised_actions_normalized']
        
                    # Step to sample from P(x_t-1 | x_t, x_0)
                    x_t_prev = self.noise_scheduler.step(
                        model_output = x_0,
                        timesteps = diffusion_steps,
                        sample = noised_action_normalized,
                        prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
                    )
                    noised_action_normalized = x_t_prev.detach()
            
            # Feed the possibly masked conditioning data into the ControlNet-conditioned denoiser
            denoise_outputs = self.forward_denoiser(
                encoder_outputs,
                noised_action_normalized,
                diffusion_steps.view(B, A),
                agents_future=control_agents_future if self._use_controlnet else None,
                agent_latents=control_agent_latents if self._use_controlnet else None,
                scenario_latents=control_scenario_latents if self._use_controlnet else None,
                text_prompt=text_prompt if self._use_controlnet else None,
                text_prompt_mask=text_prompt_mask if self._use_controlnet else None,
                replan_idx=0,
                control_mask=control_mask if self._use_controlnet else None,
            )
            
            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            if self._use_controlnet:
                film_stats = self.controlnet.get_film_stats()
                for block_name, stats in film_stats.items():
                    for key, value in stats.items():
                        log_dict[f"{prefix}film_{block_name}_{key}"] = value

            # Get Loss 
            denoised_trajs = denoise_outputs['denoised_trajs']
            if self._prediction_type == 'sample':
                state_loss_mean, yaw_loss_mean = self.denoise_loss(
                    denoised_trajs,
                    agents_future, agents_future_valid,
                    agents_interested,
                )
                denoise_loss = state_loss_mean + yaw_loss_mean 
                total_loss += denoise_loss
                
                # Predict the noise
                _, diffusion_loss = self.noise_scheduler.get_noise(
                    x_0 = denoise_outputs['denoised_actions_normalized'],
                    x_t = noised_action_normalized,
                    timesteps=diffusion_steps,
                    gt_noise=noise,
                )
                                
                log_dict.update({
                    prefix+'state_loss': state_loss_mean.item(),
                    prefix+'yaw_loss': yaw_loss_mean.item(),
                    prefix+'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean':
                pred_action_normalized = denoise_outputs['denoised_actions_normalized']
                denoise_loss = self.action_loss(
                    pred_action_normalized, gt_actions_normalized, gt_actions_valid, agents_interested
                )
                total_loss += denoise_loss
                log_dict.update({
                    prefix+'action_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')

            if self._collision_loss_weight > 0.0:
                agent_sizes = batch['agents_history'][:, :self._agents_len, -1, 5:7]
                valid_mask = (agents_future_valid > 0) & (agents_interested > 0).unsqueeze(-1)
                valid_mask = valid_mask[..., : denoised_trajs.shape[2]]
                collision_loss = self._collision_loss(denoised_trajs, agent_sizes, valid_mask)
                total_loss += self._collision_loss_weight * collision_loss
                log_dict[prefix + 'collision_loss'] = collision_loss.item()
                

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            if self._use_controlnet and prefix.startswith('val/'):
                guided_mask = control_mask > 0.5
                unguided_mask = guided_mask.logical_not()

                if guided_mask.any():
                    guided_valid = agents_future_valid & guided_mask.unsqueeze(-1)
                    guided_ade, guided_fde = self.calculate_metrics_denoise(
                        denoised_trajs, agents_future, guided_valid, agents_interested, 8
                    )
                    log_dict[f'{prefix}guided_ADE'] = guided_ade
                    log_dict[f'{prefix}guided_FDE'] = guided_fde

                if unguided_mask.any():
                    unguided_valid = agents_future_valid & unguided_mask.unsqueeze(-1)
                    unguided_ade, unguided_fde = self.calculate_metrics_denoise(
                        denoised_trajs, agents_future, unguided_valid, agents_interested, 8
                    )
                    log_dict[f'{prefix}unguided_ADE'] = unguided_ade
                    log_dict[f'{prefix}unguided_FDE'] = unguided_fde
            
            log_dict.update({
                prefix+'denoise_ADE': denoise_ade,
                prefix+'denoise_FDE': denoise_fde,
            })
        
        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss 
            goal_scores = goal_outputs['goal_scores']
            goal_trajs = goal_outputs['goal_trajs']
            
            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )

            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss += 1.0 * pred_loss 
            
            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            
            log_dict.update({
                prefix+'goal_loss': goal_loss_mean.item(),
                prefix+'score_loss': score_loss_mean.item(),
                prefix+'pred_ADE': pred_ade,
                prefix+'pred_FDE': pred_fde,
            })
        
        log_dict[prefix+'loss'] = total_loss.item()

        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    # ============================================================================
    # Closed-Loop Training Functions (Algorithm 1)
    # ============================================================================

    def generate_candidate_trajectories(
        self,
        batch: dict,
        encoder_outputs: dict,
        gt_window: torch.Tensor,
        control_agent_latents: torch.Tensor | None,
        control_scenario_latents: torch.Tensor | None,
        control_agents_future: torch.Tensor | None,
        num_candidates: int = 5,
        noise_scale: float = 1.0,
        replan_idx: Optional[int] = None,
    ):
        """
        Generate M candidate trajectories efficiently using Select-then-Recompute strategy.
        
        Optimization:
        Instead of generating M candidates with gradients (O(M) VRAM), we:
        1. Generate M candidates WITHOUT gradients (Selection Phase).
        2. Select the best candidate (Joint Scene Selection) based on GT.
        3. Re-generate ONLY the best candidate WITH gradients (Recompute Phase).
        
        This reduces VRAM usage from O(M) to O(1).

        Args:
            batch: Current batch dict
            encoder_outputs: Encoder outputs
            gt_window: Ground truth future [B, A, window_len, 5]
            ... (conditioning args)

        Returns:
            candidates: List containing ONLY the best trajectory [B, A, window_len, 5]
            diffusion_steps_list: List containing ONLY the steps for the best trajectory
        """
        B = batch['agents_history'].shape[0]
        A = self._agents_len
        device = batch['agents_history'].device
        window_len = gt_window.shape[2]

        # Prepare GT for selection
        # gt_window: [B, A, window_len, 5]
        gt_pos = gt_window[..., :2]
        gt_valid = (gt_window.sum(-1) != 0) # [B, A, window_len]

        # Prepare input actions
        # Convert GT window to actions
        gt_window_actions, gt_window_actions_valid = inverse_kinematics(
            gt_window,
            (gt_window.sum(-1) != 0),
            dt=self._dt,
            action_len=self._action_len
        )
        action_mean = torch.tensor(self._action_mean, device=device).view(1, 1, 1, 2)
        action_std = torch.tensor(self._action_std, device=device).view(1, 1, 1, 2)
        gt_window_actions_normalized = (gt_window_actions - action_mean) / action_std
        gt_window_actions_normalized = torch.clamp(
            gt_window_actions_normalized, -self._clamp_value, self._clamp_value
        )

        model_future_len = self._future_len
        num_actions_full = model_future_len // self._action_len
        num_actions_window = gt_window_actions_normalized.shape[2]
        valid_actions_len = min(num_actions_full, num_actions_window)

        # CRITICAL: Use cat instead of torch.zeros + in-place to preserve gradients!
        if valid_actions_len < num_actions_full:
            # Need padding
            padding = torch.zeros(B, A, num_actions_full - valid_actions_len, 2, device=device, dtype=gt_window_actions_normalized.dtype)
            full_actions_normalized = torch.cat([gt_window_actions_normalized[:, :, :valid_actions_len, :], padding], dim=2)
        else:
            # No padding needed
            full_actions_normalized = gt_window_actions_normalized[:, :, :valid_actions_len, :]

        # Prepare Conditioning (shared across candidates)
        agents_interested = batch['agents_interested'][:, :A]
        valid_agents = agents_interested > 0
        control_mask = valid_agents.float()
        if self._conditioning_mode == 'text':
            text_mask = batch.get('text_prompt_mask', None)
            if isinstance(text_mask, torch.Tensor):
                if text_mask.dim() == 3 and text_mask.shape[0] == 1:
                    text_mask = text_mask.squeeze(0)
                if text_mask.dim() == 1:
                    text_mask = text_mask.unsqueeze(0)
                if text_mask.shape == control_mask.shape:
                    if text_mask.dtype != torch.bool:
                        text_mask = text_mask > 0
                    control_mask = control_mask * text_mask.float()

        current_control_future = None
        if control_agents_future is not None:
            valid_len = min(model_future_len, control_agents_future.shape[2])
            # CRITICAL: Use cat instead of torch.zeros + in-place to preserve gradients!
            if valid_len < model_future_len:
                # Need padding
                padding = torch.zeros(B, A, model_future_len - valid_len, 3, device=device, dtype=control_agents_future.dtype)
                current_control_future = torch.cat([control_agents_future[:, :, :valid_len, :], padding], dim=2)
            else:
                # No padding needed
                current_control_future = control_agents_future[:, :, :valid_len, :]

        # Pre-sample Noise and Steps for M candidates
        # NEW APPROACH: Generate all M candidates in a single vectorized forward pass
        max_timestep = int(self._diffusion_steps * noise_scale)
        if max_timestep == 0: max_timestep = 1

        # Generate noise and steps for all M candidates: shape (B, M, ...)
        all_diffusion_steps = torch.randint(0, max_timestep, (B, num_candidates, A), device=device, dtype=torch.long)
        all_noise = torch.randn(B, num_candidates, *full_actions_normalized.shape[1:], device=device, dtype=full_actions_normalized.dtype)

        # Vectorized Generation: Generate all M candidates in one forward pass
        # Reshape: (B, M, ...) → (B*M, ...)
        M = num_candidates

        # Expand encoder_outputs for B*M batch
        encoder_outputs_expanded = {}
        for k, v in encoder_outputs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == B:
                # Expand first dimension: (B, ...) → (B, M, ...) → (B*M, ...)
                v_expanded = v.unsqueeze(1).expand(B, M, *v.shape[1:]).reshape(B*M, *v.shape[1:])
                encoder_outputs_expanded[k] = v_expanded
            else:
                encoder_outputs_expanded[k] = v

        # Expand conditioning
        if current_control_future is not None:
            current_control_future_expanded = current_control_future.unsqueeze(1).expand(B, M, *current_control_future.shape[1:]).reshape(B*M, *current_control_future.shape[1:])
        else:
            current_control_future_expanded = None

        if control_agent_latents is not None:
            control_agent_latents_expanded = control_agent_latents.unsqueeze(1).expand(B, M, *control_agent_latents.shape[1:]).reshape(B*M, *control_agent_latents.shape[1:])
        else:
            control_agent_latents_expanded = None

        if control_scenario_latents is not None:
            control_scenario_latents_expanded = control_scenario_latents.unsqueeze(1).expand(B, M, *control_scenario_latents.shape[1:]).reshape(B*M, *control_scenario_latents.shape[1:])
        else:
            control_scenario_latents_expanded = None

        control_text_prompt_expanded = None
        control_text_mask_expanded = None
        if self._conditioning_mode == 'text':
            control_text_prompt = batch.get('text_prompt', None)
            if control_text_prompt is not None:
                expanded_prompts = []
                if isinstance(control_text_prompt, (list, tuple)) and control_text_prompt:
                    if isinstance(control_text_prompt[0], str):
                        for _ in range(M):
                            expanded_prompts.append(list(control_text_prompt))
                    else:
                        for row in control_text_prompt:
                            for _ in range(M):
                                expanded_prompts.append(list(row))
                control_text_prompt_expanded = expanded_prompts
            control_text_mask = batch.get('text_prompt_mask', None)
            if isinstance(control_text_mask, torch.Tensor):
                if control_text_mask.dim() == 3 and control_text_mask.shape[0] == 1:
                    control_text_mask = control_text_mask.squeeze(0)
                control_text_mask_expanded = (
                    control_text_mask.unsqueeze(1)
                    .expand(B, M, *control_text_mask.shape[1:])
                    .reshape(B*M, *control_text_mask.shape[1:])
                )

        control_mask_expanded = control_mask.unsqueeze(1).expand(B, M, A).reshape(B*M, A)

        # Expand inputs
        full_actions_normalized_expanded = full_actions_normalized.unsqueeze(1).expand(B, M, *full_actions_normalized.shape[1:]).reshape(B*M, *full_actions_normalized.shape[1:])
        all_diffusion_steps_flat = all_diffusion_steps.reshape(B*M, A)
        all_noise_flat = all_noise.reshape(B*M, *all_noise.shape[2:])

        # Single vectorized forward pass for all M candidates
        noised_action_normalized_expanded = self.noise_scheduler.add_noise(
            full_actions_normalized_expanded, all_noise_flat, all_diffusion_steps_flat
        )

        denoise_outputs = self.forward_denoiser(
            encoder_outputs_expanded,
            noised_action_normalized_expanded,
            all_diffusion_steps_flat,
            agents_future=current_control_future_expanded,
            agent_latents=control_agent_latents_expanded,
            scenario_latents=control_scenario_latents_expanded,
            text_prompt=control_text_prompt_expanded if self._conditioning_mode == 'text' else None,
            text_prompt_mask=control_text_mask_expanded if self._conditioning_mode == 'text' else None,
            replan_idx=replan_idx,
            control_mask=control_mask_expanded,
            add_noise=False,
        )

        denoised_actions_normalized = denoise_outputs['denoised_actions_normalized']
        denoised_actions = denoised_actions_normalized * action_std + action_mean
        current_states = encoder_outputs_expanded['agents'][:, :A, -1, :5]
        denoised_trajs = roll_out(
            current_states=current_states,
            actions=denoised_actions,
            dt=self._dt,
            action_len=self._action_len,
            global_frame=False,
            add_noise=False
        )
        denoised_trajs = batch_transform_trajs_to_global_frame(denoised_trajs, current_states)

        # Reshape back: (B*M, A, T, 5) → (B, M, A, T, 5)
        denoised_trajs = denoised_trajs.reshape(B, M, A, -1, 5)

        # Selection: Compute ADE for each candidate and select best
        gen_len = denoised_trajs.shape[3]
        target_len = gt_pos.shape[2]
        common_len = min(gen_len, target_len)

        candidates_window = denoised_trajs[:, :, :, :common_len, :]  # (B, M, A, common_len, 5)
        candidates_pos = candidates_window[..., :2]  # (B, M, A, common_len, 2)
        gt_pos_cropped = gt_pos[:, :, :common_len, :].unsqueeze(1)  # (B, 1, A, common_len, 2)

        # Compute ADE: detach only for selection criterion (not the candidates themselves!)
        diff = torch.norm(candidates_pos.detach() - gt_pos_cropped, dim=-1)  # (B, M, A, common_len)
        gt_valid_cropped = gt_valid[:, :, :common_len].unsqueeze(1)  # (B, 1, A, common_len)
        mask = gt_valid_cropped & (agents_interested.unsqueeze(1).unsqueeze(-1) > 0)

        masked_diff = diff * mask
        error_sum = masked_diff.sum(dim=(2, 3))  # (B, M)
        valid_count = mask.sum(dim=(2, 3))  # (B, M)
        joint_ade = error_sum / (valid_count + 1e-6)  # (B, M)

        # Select best candidate per batch
        best_indices = joint_ade.argmin(dim=1)  # (B,)

        # Advanced indexing to select best (PRESERVES GRADIENTS!)
        batch_indices = torch.arange(B, device=device)
        best_candidate_window = candidates_window[batch_indices, best_indices, :, :window_len, :]  # (B, A, window_len, 5)

        # Collect final diffusion steps for compatibility
        final_diffusion_steps = all_diffusion_steps[batch_indices, best_indices]  # (B, A)

        return [best_candidate_window], [final_diffusion_steps]

    def select_best_candidate(
        self,
        candidates: list,
        gt_segment: torch.Tensor,
        segment_valid: torch.Tensor,
    ):
        # Since we moved selection inside generate_candidate_trajectories for efficiency,
        # this function now just returns the single candidate provided (which is already the best).
        # We keep the signature for compatibility or simple pass-through.
        return candidates[0], None # indices not needed as we already selected


    def forward_and_get_loss_closed_loop(
        self,
        batch: dict,
        segment_len: int = 16, # Deprecated in favor of execution_horizon
        execution_horizon: int = 11,
        prediction_horizon: int = 40,
        num_candidates: int = 5,
        noise_scale: float = 1.0,
        prefix: str = 'train/',
    ):
        """
        Closed-loop training forward pass using Receding Horizon Control (Sliding Window).

        Algorithm:
        1. Initialize current_time t = 0.
        2. While t + prediction_horizon <= total_future_len:
           a. Define prediction window: [t : t + prediction_horizon]
           b. Extract GT window and Conditioning for this window.
           c. Generate M candidates for this window.
           d. Select best candidate based on window ADE.
           e. Compute loss on the FULL prediction window.
           f. Execute the first `execution_horizon` steps of the best candidate.
           g. Update agent history and map relations.
           h. Advance t += execution_horizon.

        Args:
            batch: Input batch dict
            execution_horizon: Steps to execute before re-planning (e.g., 11)
            prediction_horizon: Steps to predict in each plan (e.g., 40)
            num_candidates: Number of candidates M to generate (default 5)
            noise_scale: γ in Algorithm 1 (default 1.0)
            prefix: Logging prefix

        Returns:
            loss: Total loss
            log_dict: Logging dictionary
        """
        # Use correct parameters (handle legacy call signature if needed)
        exec_h = execution_horizon
        pred_h = prediction_horizon
        

        B = batch['agents_history'].shape[0]
        A = self._agents_len
        device = batch['agents_history'].device

        # Get GT future
        agents_future = batch['agents_future'][:, :A]  # [B, A, 80, 8]
        agents_future_valid = (agents_future.sum(-1) != 0).float()  # [B, A, 80]
        agents_interested = batch['agents_interested'][:, :A]
        total_future_len = agents_future.shape[2]

        # Prepare conditioning
        agent_latents_segments = None
        agent_latents_single = None
        
        if self._conditioning_mode == 'latent':
            if 'agent_latents_segments' in batch:
                agent_latents_segments = batch['agent_latents_segments'][:, :, :A].clone()
            else:
                # Fallback to single latent
                agent_latents_single = batch.get('agent_latents', None)
                if agent_latents_single is not None:
                    agent_latents_single = agent_latents_single[:, :A].clone()

        # Initialize loop variables
        current_time = 0
        total_loss = 0.0
        step_count = 0
        
        segment_losses = []
        segment_state_losses = []
        segment_yaw_losses = []
        segment_collision_losses = []

        # Create working copy of batch for updates
        working_batch = {}
        # List of all agent-related fields that need slicing along dimension 1
        agent_fields = ['agents_history', 'agents_future', 'agents_interested',
                       'agents_type', 'anchors', 'agent_ids']
        # NEW: Segment latents also need slicing along agent dimension
        segment_latent_fields = ['agent_latents_segments']

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k in agent_fields:
                    working_batch[k] = v[:, :A].clone()
                elif k in segment_latent_fields:
                    working_batch[k] = v[:, :, :A].clone()
                else:
                    working_batch[k] = v.clone()
            else:
                working_batch[k] = v

        # Recalculate initial relations
        from ..model.model_utils import calculate_relations_torch
        working_batch['relations'] = calculate_relations_torch(
            working_batch['agents_history'],
            working_batch['polylines'],
            working_batch['traffic_light_points']
        )

        # ========== Receding Horizon Loop ==========
        # Continue as long as we have enough GT for a full prediction window
        while current_time + pred_h <= total_future_len:
            window_end = current_time + pred_h

            # Re-encode scene with updated history
            encoder_outputs = self.encoder(working_batch)

            # Extract GT Window (Extended to max available context to minimize zero-padding)
            # This helps the Transformer see a continuous future if available, rather than sudden zeros.
            max_window_end = min(current_time + self._future_len, total_future_len)
            
            # Ensure window length satisfies inverse_kinematics constraint: (len - 1) % action_len == 0
            current_window_len = max_window_end - current_time
            remainder = (current_window_len - 1) % self._action_len
            if remainder != 0:
                max_window_end -= remainder
            
            gt_window_extended = agents_future[:, :, current_time:max_window_end, :5]
            # gt_window_valid_extended = agents_future_valid[:, :, current_time:max_window_end]

            # Prepare Conditioning for this window
            current_agent_latents = None
            current_control_future = None

            if self._use_controlnet:
                if self._conditioning_mode == 'latent':
                    latent_seg_len = int(getattr(self, "_latent_segment_len", pred_h))
                    if latent_seg_len <= 0:
                        latent_seg_len = pred_h
                    
                    if agent_latents_segments is not None:
                        # Clamp seg_idx to available segments
                        num_segs = agent_latents_segments.shape[1]
                        if getattr(self, "_latent_index_mode", "window") == "time":
                            seg_idx = current_time // latent_seg_len
                        else:
                            # Default: segment index is the planning window index (aligned with execution_horizon shifts).
                            seg_idx = step_count
                        seg_idx = min(int(seg_idx), num_segs - 1)
                        
                        current_agent_latents = agent_latents_segments[:, seg_idx]
                    elif agent_latents_single is not None:
                        current_agent_latents = agent_latents_single
                
                elif self._conditioning_mode == 'future':
                    if self._future_conditioning_mode == 'static_full':
                        current_control_future = agents_future[:, :, :self._future_len, :3].clone()
                    else:
                        # Strategy: Extract relevant window from future trajectory sketch
                        current_control_future = agents_future[:, :, current_time:window_end, :3].clone() # [B, A, pred_h, 3]

                # Apply dropout (re-sample per window or keep consistent? Standard is per-step in training)
                # For simplicity and robustness, we re-apply dropout logic here
                if self.training and self._per_agent_dropout_max > 0.0:
                    # Reuse logic from forward_and_get_loss (simplified)
                    dropout_prob = torch.empty(1, device=device).uniform_(
                        self._per_agent_dropout_min, self._per_agent_dropout_max
                    ).item()
                    if dropout_prob > 0.0:
                        # Create mask [B, A]
                        random_tensor = torch.rand((B, A), device=device)
                        dropout_mask = (random_tensor < dropout_prob) & (agents_interested > 0)
                        # Ensure one valid
                        for b in range(B):
                            if (agents_interested[b] > 0).any():
                                if (dropout_mask[b] & (agents_interested[b]>0)).all():
                                    idx = torch.nonzero(agents_interested[b]>0, as_tuple=False)[0]
                                    dropout_mask[b, idx] = False
                        
                        if current_control_future is not None:
                            current_control_future = current_control_future.masked_fill(
                                dropout_mask.unsqueeze(-1).unsqueeze(-1), 0.0
                            )
                        if current_agent_latents is not None:
                            current_agent_latents = current_agent_latents.masked_fill(
                                dropout_mask.unsqueeze(-1), 0.0
                            )

            # Generate Candidates (using extended window)
            candidates_extended, _ = self.generate_candidate_trajectories(
                batch=working_batch,
                encoder_outputs=encoder_outputs,
                gt_window=gt_window_extended,
                control_agent_latents=current_agent_latents,
                control_scenario_latents=None,
                control_agents_future=current_control_future,
                num_candidates=num_candidates,
                noise_scale=noise_scale,
                replan_idx=step_count,
            )

            # Slice candidates and GT back to prediction_horizon for Evaluation/Selection
            candidates = [c[:, :, :pred_h, :] for c in candidates_extended]

            gt_window_target = agents_future[:, :, current_time:window_end, :5] # [B, A, pred_h, 5]
            gt_window_valid_target = agents_future_valid[:, :, current_time:window_end] # [B, A, pred_h]

            # Select Best Candidate based on TARGET window ADE
            best_window, _ = self.select_best_candidate(
                candidates=candidates,
                gt_segment=gt_window_target,
                segment_valid=gt_window_valid_target,
            )




            # Compute Loss on full prediction window (using target horizon)
            win_state_loss, win_yaw_loss = self.denoise_loss(
                best_window,
                gt_window_target,
                gt_window_valid_target,
                agents_interested,
                skip_first_frame=False
            )

            step_loss = win_state_loss + win_yaw_loss
            if self._collision_loss_weight > 0.0:
                agent_sizes = working_batch['agents_history'][:, :A, -1, 5:7]
                valid_mask = (gt_window_valid_target > 0) & (agents_interested > 0).unsqueeze(-1)
                valid_mask = valid_mask[..., : best_window.shape[2]]
                collision_loss = self._collision_loss(best_window, agent_sizes, valid_mask)
                step_loss = step_loss + self._collision_loss_weight * collision_loss
                segment_collision_losses.append(collision_loss.item())

            total_loss += step_loss
            step_count += 1
            
            segment_losses.append(step_loss.item())
            segment_state_losses.append(win_state_loss.item())
            segment_yaw_losses.append(win_yaw_loss.item())

            # Prepare for next step: Extract executed portion
            # FIXED: Improved Teacher Forcing Strategy with Scheduled Sampling
            # Old issues:
            # 1. Random 50% batch-level dropout was too aggressive (error accumulation)
            # 2. Per-agent random masking broke agent interaction learning
            # 3. No curriculum learning (model had to handle both extremes from start)
            #
            # New strategy: Curriculum learning with scheduled sampling
            # - Start with high GT ratio (0.9) to learn stable dynamics
            # - Gradually decay to low ratio (0.3) to adapt to closed-loop
            # - Batch-level decision to maintain consistent agent interactions
            # - Detach is necessary to prevent gradient graph explosion across windows

            # NOTE: detach() is REQUIRED here because:
            # 1. Loss is already computed on best_window (line 1494-1500)
            # 2. executed_traj is only for updating history, not for computing loss
            # 3. Without detach, gradients would accumulate across entire receding horizon
            gt_exec = gt_window_target[:, :, :exec_h, :].detach()
            pred_exec = best_window[:, :, :exec_h, :].detach()
            
            actual_tf_ratio = 0.0

            if self.training:
                # Dynamic Curriculum: 0.9 -> 0.5 over 40 epochs
                current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
                tf_prob = max(0.5, 0.9 - current_epoch / 100.0)
                
                apply_tf = (torch.rand(1).item() < tf_prob)
                
                if apply_tf:
                    # Create mask: 1 for GT, 0 for Pred
                    # 70% of agents use GT (Paper setting)
                    tf_mask = (torch.rand(B, A, device=device) < 0.7).float()
                else:
                    # 0% use GT (All Prediction)
                    tf_mask = torch.zeros(B, A, device=device)
                
                # Record actual ratio for logging
                actual_tf_ratio = tf_mask.mean().item()

                # Expand mask for broadcasting: [B, A] -> [B, A, exec_h, 5]
                tf_mask = tf_mask.view(B, A, 1, 1)
                executed_traj = tf_mask * gt_exec + (1 - tf_mask) * pred_exec

            else:
                # Validation/Testing: Always use Prediction (Pure Closed-Loop)
                executed_traj = pred_exec
                actual_tf_ratio = 0.0
            
            # Update working batch history
            from ..model.model_utils import update_batch_for_training
            working_batch = update_batch_for_training(
                batch=working_batch,
                predicted_segment=executed_traj,
            )

            # Advance time
            current_time += exec_h

        # Average loss
        if step_count > 0:
            total_loss = total_loss / step_count
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Logging
        log_dict = {
            prefix + 'total_loss': total_loss.item(),
            prefix + 'state_loss': sum(segment_state_losses) / len(segment_state_losses) if segment_state_losses else 0.0,
            prefix + 'yaw_loss': sum(segment_yaw_losses) / len(segment_yaw_losses) if segment_yaw_losses else 0.0,
            prefix + 'num_windows': step_count,
            prefix + 'teacher_forcing_ratio': actual_tf_ratio,
        }
        if segment_collision_losses:
            log_dict[prefix + 'collision_loss'] = sum(segment_collision_losses) / len(segment_collision_losses)

        # Add FiLM statistics if using ControlNet
        if self._use_controlnet:
            film_stats = self.controlnet.get_film_stats()
            for block_name, stats in film_stats.items():
                for key, value in stats.items():
                    log_dict[f"{prefix}film_{block_name}_{key}"] = value

        return total_loss, log_dict

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

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Supports both open-loop and closed-loop training:
        - Open-loop: Standard diffusion training (forward_and_get_loss)
        - Closed-loop: Receding horizon with candidate generation (forward_and_get_loss_closed_loop)

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """
        # Check if closed-loop training is enabled
        use_closed_loop = getattr(self, '_use_closed_loop_training', False)

        if use_closed_loop:
            # Closed-loop training (Algorithm 1)
            execution_horizon = getattr(self, '_closed_loop_execution_horizon', 11)
            prediction_horizon = getattr(self, '_closed_loop_prediction_horizon', 40)
            num_candidates = getattr(self, '_closed_loop_num_candidates', 5)
            noise_scale = getattr(self, '_closed_loop_noise_scale', 1.0)

            loss, log_dict = self.forward_and_get_loss_closed_loop(
                batch=batch,
                execution_horizon=execution_horizon,
                prediction_horizon=prediction_horizon,
                num_candidates=num_candidates,
                noise_scale=noise_scale,
                prefix='train/',
            )
        else:
            # Open-loop training (standard)
            loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')

        self.log_dict(
            log_dict,
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )

        if (not torch.is_tensor(loss)) or (not loss.requires_grad):
            self.log(
                'train/skip_no_grad',
                1,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=1,
            )
            return self._zero_loss_with_grad()

        return loss

    def on_after_backward(self):
        """
        Hook called immediately after backward().
        """
        pass

    
    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Supports both open-loop and closed-loop validation:
        - Open-loop: Standard diffusion validation (forward_and_get_loss)
        - Closed-loop: Receding horizon validation (forward_and_get_loss_closed_loop)

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """
        # Check if closed-loop training is enabled
        use_closed_loop = getattr(self, '_use_closed_loop_training', False)

        if use_closed_loop:
            # Closed-loop validation
            execution_horizon = getattr(self, '_closed_loop_execution_horizon', 11)
            prediction_horizon = getattr(self, '_closed_loop_prediction_horizon', 40)
            num_candidates = getattr(self, '_closed_loop_num_candidates', 5)
            noise_scale = getattr(self, '_closed_loop_noise_scale', 1.0)

            loss, log_dict = self.forward_and_get_loss_closed_loop(
                batch=batch,
                execution_horizon=execution_horizon,
                prediction_horizon=prediction_horizon,
                num_candidates=num_candidates,
                noise_scale=noise_scale,
                prefix='val/',
            )
        else:
            # Open-loop validation
            loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')

        self.log_dict(log_dict,
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)

        return loss

    ################### Loss function ###################
    def denoise_loss(
            self, denoised_trajs,
            agents_future, agents_future_valid,
            agents_interested,
            skip_first_frame=True
        ):
            """
            Calculates the denoise loss for the denoised actions and trajectories.

            Args:
                denoised_actions_normalized (torch.Tensor): Normalized denoised actions tensor of shape [B, A, T, C].
                denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
                agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
                agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
                gt_actions_normalized (torch.Tensor): Normalized ground truth actions tensor of shape [B, A, T, C].
                gt_actions_valid (torch.Tensor): Ground truth actions validity tensor of shape [B, A, T].
                agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].
                skip_first_frame (bool): If True, skip the first timestep (for open-loop).
                                        If False, use all timesteps (for closed-loop segments).

            Returns:
                state_loss_mean (torch.Tensor): Mean state loss.
                yaw_loss_mean (torch.Tensor): Mean yaw loss.
                action_loss_mean (torch.Tensor): Mean action loss.
            """

            # For closed-loop training, we don't skip the first frame since segments
            # are already extracted from the middle of the trajectory
            if skip_first_frame:
                agents_future = agents_future[..., 1:, :3]
                future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)
            else:
                agents_future = agents_future[..., :3]
                future_mask = agents_future_valid * (agents_interested[..., None] > 0)

            # Calculate State Loss
            # [B, A, T]
            state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(-1)
            yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_loss = torch.abs(yaw_error)
            
            # Filter out the invalid state
            state_loss = state_loss * future_mask
            yaw_loss = yaw_loss * future_mask

            mask_sum = future_mask.sum()
            if mask_sum.item() == 0:
                zero = denoised_trajs.sum() * 0.0
                return zero, zero

            # Calculate the mean loss
            state_loss_mean = state_loss.sum() / mask_sum
            yaw_loss_mean = yaw_loss.sum() / mask_sum
            
            return state_loss_mean, yaw_loss_mean

    def _collision_loss(
        self,
        trajs: torch.Tensor,  # [B,A,T,>=3]
        agent_sizes: torch.Tensor,  # [B,A,2] (length, width)
        valid_mask: torch.Tensor,  # [B,A,T] bool
    ) -> torch.Tensor:
        if trajs.dim() != 4 or trajs.shape[-1] < 3:
            return trajs.sum() * 0.0
        if agent_sizes.dim() != 3 or agent_sizes.shape[-1] < 2:
            return trajs.sum() * 0.0

        b, a, t, _ = trajs.shape
        if a < 2:
            return trajs.sum() * 0.0

        length = agent_sizes[..., 0].clamp(min=1e-3)
        width = agent_sizes[..., 1].clamp(min=1e-3)
        radius = 0.5 * torch.minimum(length, width)  # circle stays inside length/width
        half_len = 0.5 * length
        offset = torch.stack(
            [-(half_len - radius), torch.zeros_like(radius), (half_len - radius)], dim=-1
        )  # [B,A,3]

        theta = trajs[..., 2]
        cos_t = torch.cos(theta).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        pos = trajs[..., :2]
        offset_t = offset.unsqueeze(2).expand(b, a, t, 3)
        cx = pos[..., 0].unsqueeze(-1) + offset_t * cos_t
        cy = pos[..., 1].unsqueeze(-1) + offset_t * sin_t
        centers = torch.stack([cx, cy], dim=-1)  # [B,A,T,3,2]

        ci = centers.unsqueeze(2).unsqueeze(5)  # [B,A,1,T,3,1,2]
        cj = centers.unsqueeze(1).unsqueeze(4)  # [B,1,A,T,1,3,2]
        dist = torch.norm(ci - cj, dim=-1)  # [B,A,A,T,3,3]
        min_dist = dist.amin(dim=(-1, -2))  # [B,A,A,T]

        rsum = (radius.unsqueeze(2) + radius.unsqueeze(1)).unsqueeze(-1)  # [B,A,A,1]
        overlap = torch.relu(rsum - min_dist)

        pair_mask = valid_mask.to(torch.bool)
        pair_mask = pair_mask.unsqueeze(2) & pair_mask.unsqueeze(1)  # [B,A,A,T]
        upper = torch.triu(torch.ones(a, a, device=trajs.device, dtype=torch.bool), diagonal=1)
        pair_mask = pair_mask & upper.unsqueeze(0).unsqueeze(-1)

        denom = pair_mask.sum().clamp(min=1)
        return (overlap * pair_mask.float()).sum() / denom

        
    def action_loss(
        self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        
        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask

        mask_sum = action_mask.sum()
        if mask_sum.item() == 0:
            return actions.sum() * 0.0

        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / mask_sum

        return action_loss_mean
    
    def goal_loss(
        self, trajs, scores, agents_future,
        agents_future_valid, anchors,
        agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B*A, Q, T, 3] representing predicted trajectories.
            scores (torch.Tensor): Tensor of shape [B*A, Q] representing predicted scores.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            anchors (torch.Tensor): Tensor of shape [B, A, Q, 2] representing anchor points.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        # Convert Anchor to Global Frame
        current_states = agents_future[:, :, 0, :3] 
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape
        
        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0) # [B, A, T]
        
        # Flatten batch and agents
        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1) # [B*A, 1, 2]
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1) # [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3] # [B*A, Q, T, 3]
        anchors_global = anchors_global.flatten(0, 1) # [B*A, Q, 2]
        
        # Find the closest anchor
        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1) # [B*A,]
        
        # For agents that do not have valid end point, use the minADE
        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1) # [B*A, Q, T]
        dist = dist * traj_mask.flatten(0, 1)[:, None, :] # [B*A, Q, T]
        idx = torch.argmin(dist.mean(-1), dim=-1) # [B*A,]

        # Select trajectory
        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch*num_agents), idx] # [B*A, T, 3]
        
        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction='none').sum(-1) # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1) # [B*A, T]

        traj_mask_sum = traj_mask.sum()
        interested_mask = (agents_interested > 0)
        interested_sum = interested_mask.sum()

        if traj_mask_sum.item() == 0 or interested_sum.item() == 0:
            zero = trajs.sum() * 0.0
            return zero, zero

        # Calculate the score loss
        scores = scores.flatten(0, 1) # [B*A, Q]
        score_loss = cross_entropy(scores, idx, reduction='none') # [B*A]
        score_loss = score_loss * interested_mask.flatten(0, 1) # [B*A]

        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask_sum
        score_loss_mean = score_loss.sum() / interested_sum

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self, 
            denoised_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the denoising metrics for the predicted trajectories.

            Args:
                denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
                agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
                agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
                agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
                top_k (int, optional): Number of top agents to consider. Defaults to None.

            Returns:
                Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
            """
            
            if not top_k:
                top_k = self._agents_len  
            
            pred_traj = denoised_trajs[:, :top_k, :, :2] # [B, A, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 

            valid_count = gt_mask.sum()
            if valid_count.item() == 0:
                return 0.0, 0.0

            denoise_mse = torch.norm(pred_traj - gt, dim = -1)
            denoise_ADE = denoise_mse[gt_mask].mean()

            final_mask = gt_mask[..., -1]
            if final_mask.sum().item() == 0:
                denoise_FDE = denoise_ADE
            else:
                denoise_FDE = denoise_mse[..., -1][final_mask].mean()
            
            return denoise_ADE.item(), denoise_FDE.item()
    
    @torch.no_grad()
    def calculate_metrics_predict(self,
            goal_trajs, agents_future, agents_future_valid,
            agents_interested, top_k = None
        ):
            """
            Calculates the metrics for predicting goal trajectories.

            Args:
                goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
                agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
                agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
                agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
                top_k (int, optional): The number of top agents to consider. Defaults to None.

            Returns:
                tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
            """
            
            if not top_k:
                top_k = self._agents_len
            goal_trajs = goal_trajs[:, :top_k, :, :, :2] # [B, A, Q, T, 2]
            gt = agents_future[:, :top_k, 1:, :2] # [B, A, T, 2]
            gt_mask = (agents_future_valid[:, :top_k, 1:] \
                & (agents_interested[:, :top_k, None] > 0)).bool() # [B, A, T] 
                   
            valid_count = gt_mask.sum()
            if valid_count.item() == 0:
                return 0.0, 0.0

            goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim = -1) # [B, A, Q, T]
            goal_mse = goal_mse * gt_mask[..., None, :] # [B, A, Q, T]
            best_idx = torch.argmin(goal_mse.sum(-1), dim = -1) 
            
            best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
                                     torch.arange(goal_mse.shape[1])[None, :],
                                     best_idx]
            
            goal_ADE = best_goal_mse.sum() / valid_count
            final_mask = gt_mask[..., -1]
            if final_mask.sum().item() == 0:
                goal_FDE = goal_ADE
            else:
                goal_FDE = best_goal_mse[..., -1].sum()/final_mask.sum()
            
            return goal_ADE.item(), goal_FDE.item()
    
    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        """
        Normalize the given actions using the mean and standard deviation.

        Args:
            actions : The actions to be normalized.

        Returns:
            The normalized actions.
        """
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_actions(self, actions: torch.Tensor):
        """
        Unnormalize the given actions using the stored action standard deviation and mean.

        Args:
            actions: The normalized actions to be unnormalized.

        Returns:
             The unnormalized actions.
        """
        return actions * self.action_std + self.action_mean
    
