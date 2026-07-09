import torch
import torch.nn as nn
import lightning.pytorch as pl
import math
from typing import List, Optional, Iterable, Tuple
from torch.nn.functional import smooth_l1_loss, cross_entropy

# ============================================================================
# Temporal Encoder Components
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

    def forward(self, agents_future, return_sequence: bool = False):
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

        if return_sequence:
            return x  # [B, A, T, feature_dim]

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
                    raise ValueError(
                        f"scenario_latents must be [B,{self.scenario_latent_dim}], got {scenario_latents.shape}"
                    )
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
        # Pooler output is unused; keep it frozen to avoid DDP unused param errors.
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

        # Initialize projection gently.
        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def _init_weights(self) -> None:
        # Keep compatibility with control encoder init hook.
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
        # fallback to CLS token
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


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning.

    Uses direct FiLM without scaling, with configurable initialization.
    Default is identity (gamma=1, beta=0).
    """

    def __init__(
        self,
        feature_dim,
        conditioning_dim,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        init_mode: str = "identity",
        identity_noise_std: float = 0.001,
    ):
        super().__init__()
        self.feature_dim = feature_dim # feature is the query from denoiser
        self.conditioning_dim = conditioning_dim
        self.init_mode = str(init_mode).lower()
        if self.init_mode not in {"identity", "random"}:
            raise ValueError(
                f"FiLMLayer.init_mode must be 'identity' or 'random', got {init_mode}"
            )
        self.identity_noise_std = float(identity_noise_std)
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
        # Initialize with a configurable mode for ablations.
        with torch.no_grad():
            if self.init_mode == "identity":
                nn.init.xavier_uniform_(self.film_mlp[-1].weight, gain=0.01)
                self.film_mlp[-1].bias[:self.feature_dim].fill_(1.0)  # gamma starts around 1
                self.film_mlp[-1].bias[self.feature_dim:].fill_(0.0)  # beta starts around 0
                if self.identity_noise_std > 0.0:
                    # Optional tiny perturbation to avoid exact identity.
                    self.film_mlp[-1].bias.add_(
                        torch.randn_like(self.film_mlp[-1].bias) * self.identity_noise_std
                    )
            else:  # random
                nn.init.xavier_uniform_(self.film_mlp[-1].weight, gain=1.0)
                self.film_mlp[-1].bias.zero_()

    def forward(self, x, conditioning):
        """
        Args:
            x: [B, A, T, F] - denoiser query input features
            conditioning: [B, A, C] or [B, A, T, C] - conditioning features
        Returns:
            modulated_x: [B, A, T, F] - FiLM modulated features
        """
        B, A, T, F = x.shape
        if conditioning.dim() == 3:
            # Expand conditioning to match temporal dimension
            conditioning_expanded = conditioning.unsqueeze(2).expand(B, A, T, -1)  # [B, A, T, C]
        elif conditioning.dim() == 4:
            conditioning_expanded = conditioning
        else:
            raise ValueError(
                f"conditioning must be [B,A,C] or [B,A,T,C], got {conditioning.shape}"
            )
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


class AdditiveLayer(nn.Module):
    """Additive conditioning layer.

    Applies direct additive fusion: x + delta(c), initialized near identity
    via delta(c) ~= 0.
    """

    def __init__(
        self,
        feature_dim,
        conditioning_dim,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim
        self._last_stats = {}

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
        mlp_layers.append(nn.Linear(hidden_dim, feature_dim))
        self.add_mlp = nn.Sequential(*mlp_layers)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.add_mlp[-1].weight, gain=0.01)
            self.add_mlp[-1].bias.zero_()
            self.add_mlp[-1].bias.add_(torch.randn_like(self.add_mlp[-1].bias) * 0.001)

    def forward(self, x, conditioning):
        """
        Args:
            x: [B, A, T, F] - denoiser query input features
            conditioning: [B, A, C] or [B, A, T, C] - conditioning features
        Returns:
            fused_x: [B, A, T, F] - additively fused features
        """
        B, A, T, F = x.shape
        if conditioning.dim() == 3:
            conditioning_expanded = conditioning.unsqueeze(2).expand(B, A, T, -1)
        elif conditioning.dim() == 4:
            conditioning_expanded = conditioning
        else:
            raise ValueError(
                f"conditioning must be [B,A,C] or [B,A,T,C], got {conditioning.shape}"
            )

        conditioning_flat = conditioning_expanded.reshape(B * A * T, -1)
        delta = self.add_mlp(conditioning_flat).reshape(B, A, T, F)

        with torch.no_grad():
            delta_detached = delta.detach()
            self._last_stats = {
                "delta_mean": delta_detached.mean().item(),
                "delta_std": delta_detached.std(unbiased=False).item(),
                "delta_min": delta_detached.min().item(),
                "delta_max": delta_detached.max().item(),
                "delta_abs_mean": delta_detached.abs().mean().item(),
            }

        return x + delta

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
        film_init_mode: str = "identity",
        film_identity_noise_std: float = 0.001,
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
    ):
        super().__init__()
        self._future_feature_dim = future_feature_dim
        self._agents_len = agents_len
        self._film_blocks = film_blocks
        self._conditioning_mode = conditioning_mode

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
            self.tag_encoder = None
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
            self.tag_encoder = None
        else:
            raise ValueError(
                f"Unknown conditioning_mode: {conditioning_mode}. Must be 'future' or 'latent'."
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
                init_mode=film_init_mode,
                identity_noise_std=film_identity_noise_std,
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

        else:
            raise ValueError(f"Unknown conditioning_mode: {self._conditioning_mode}")

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
 
