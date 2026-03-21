import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .config import JEPAWorldModelConfig


# ---------------------------------------------------------------------------
# Input Adapter: RMSNorm + Linear + residual
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class InputAdapter(nn.Module):
    """RMSNorm + Linear + residual adapter for distribution alignment."""

    def __init__(self, dim: int, adapter_type: str = "rmsnorm_linear_residual",
                 bottleneck_dim: int = 256):
        super().__init__()
        self.norm = RMSNorm(dim)
        if adapter_type == "bottleneck":
            self.linear = nn.Sequential(
                nn.Linear(dim, bottleneck_dim, bias=False),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, dim, bias=False),
            )
        else:
            self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# Legacy: Independent Transformer Predictor (predictor_type == "independent_transformer")
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Bidirectional Transformer predictor with learned prediction queries.

    This is the legacy independent predictor kept for ablation comparison.
    """

    def __init__(self, config: JEPAWorldModelConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.n_visual_tokens = config.n_visual_tokens
        self.max_action_tokens = config.max_action_tokens

        self.prediction_queries = nn.Parameter(
            torch.empty(1, config.n_visual_tokens, hidden_dim),
        )
        nn.init.trunc_normal_(
            self.prediction_queries,
            std=1.0 / math.sqrt(hidden_dim),
        )

        self.use_learned_positional_embeddings = (
            config.use_learned_positional_embeddings
        )
        if self.use_learned_positional_embeddings:
            self.state_pos = nn.Parameter(
                torch.zeros(1, config.n_visual_tokens, hidden_dim),
            )
            self.action_pos = nn.Parameter(
                torch.zeros(1, config.max_action_tokens, hidden_dim),
            )
            self.query_pos = nn.Parameter(
                torch.zeros(1, config.n_visual_tokens, hidden_dim),
            )
            nn.init.normal_(self.state_pos, std=0.02)
            nn.init.normal_(self.action_pos, std=0.02)
            nn.init.normal_(self.query_pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_predictor_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_state_tokens, _ = state_tokens.shape
        num_action_tokens = action_tokens.shape[1]

        if num_state_tokens != self.n_visual_tokens:
            raise ValueError(
                f"Expected {self.n_visual_tokens} state tokens, got {num_state_tokens}.",
            )
        if num_action_tokens > self.max_action_tokens:
            raise ValueError(
                f"Action token length {num_action_tokens} exceeds "
                f"max_action_tokens={self.max_action_tokens}.",
            )

        prediction_queries = self.prediction_queries.expand(batch_size, -1, -1)

        if self.use_learned_positional_embeddings:
            state_tokens = state_tokens + self.state_pos[:, :num_state_tokens]
            action_tokens = action_tokens + self.action_pos[:, :num_action_tokens]
            prediction_queries = prediction_queries + self.query_pos[:, : self.n_visual_tokens]

        sequence = torch.cat(
            [state_tokens, action_tokens, prediction_queries],
            dim=1,
        )

        padding_mask = None
        if action_padding_mask is not None:
            state_mask = torch.zeros(
                batch_size, num_state_tokens,
                device=sequence.device, dtype=torch.bool,
            )
            query_mask = torch.zeros(
                batch_size, self.n_visual_tokens,
                device=sequence.device, dtype=torch.bool,
            )
            padding_mask = torch.cat(
                [state_mask, action_padding_mask.to(sequence.device), query_mask],
                dim=1,
            )

        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)
        predicted = encoded[:, num_state_tokens + num_action_tokens :, :]
        return self.output_norm(predicted)


# ---------------------------------------------------------------------------
# New: Qwen Bidirectional Predictor with LoRA
# ---------------------------------------------------------------------------

class QwenBidirectionalPredictor(nn.Module):
    """Predictor using Qwen2.5-VL LLM backbone with LoRA and bidirectional attention.

    Architecture:
        state_adapter(state_tokens)  + token_type/spatial/time embeds
        action_adapter(action_embed) + token_type/time embeds
        query_adapter(query_tokens)  + token_type/spatial/time embeds
            → concat (B, 19, D)
            → Qwen LLM 36 layers (frozen + LoRA on q/k/v/o, RMSNorm unfrozen)
            → bidirectional attention (no causal mask)
            → RoPE preserved (position_ids = 0..18)
            → extract query positions → output_norm
            → predicted_tokens (B, 9, D)
    """

    def __init__(self, config: JEPAWorldModelConfig, llm_layers: nn.ModuleList):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        N = config.n_visual_tokens

        # --- Learned prediction queries ---
        self.prediction_queries = nn.Parameter(torch.empty(1, N, D))
        nn.init.trunc_normal_(self.prediction_queries, std=1.0 / math.sqrt(D))

        # --- Input adapters ---
        if config.use_input_adapters:
            self.state_adapter = InputAdapter(
                D, config.adapter_type, config.adapter_bottleneck_dim,
            )
            self.action_adapter = InputAdapter(
                D, config.adapter_type, config.adapter_bottleneck_dim,
            )
            self.query_adapter = InputAdapter(
                D, config.adapter_type, config.adapter_bottleneck_dim,
            )
        else:
            self.state_adapter = nn.Identity()
            self.action_adapter = nn.Identity()
            self.query_adapter = nn.Identity()

        # --- Learned embeddings ---
        # token_type: 0=state, 1=action, 2=query
        if config.use_token_type_embed:
            self.token_type_embed = nn.Embedding(3, D)
            nn.init.normal_(self.token_type_embed.weight, std=0.02)

        # spatial: 3x3 grid positions (shared between state and query)
        if config.use_spatial_embed:
            self.spatial_embed = nn.Embedding(N, D)
            nn.init.normal_(self.spatial_embed.weight, std=0.02)

        # time: 0=current (state+action), 1=future (query)
        if config.use_time_embed:
            self.time_embed = nn.Embedding(2, D)
            nn.init.normal_(self.time_embed.weight, std=0.02)

        # --- Copy Qwen LLM decoder layers (deep copy, bidirectional) ---
        n_layers = min(config.n_llm_layers, len(llm_layers))
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = copy.deepcopy(llm_layers[i])
            self.layers.append(layer)

        # Freeze all base weights, then selectively unfreeze
        for param in self.layers.parameters():
            param.requires_grad_(False)

        # Unfreeze all RMSNorm / LayerNorm parameters in the copied layers
        if config.lora.unfreeze_norms:
            for layer in self.layers:
                for name, module in layer.named_modules():
                    if "norm" in name.lower() or "layernorm" in name.lower():
                        for param in module.parameters():
                            param.requires_grad_(True)

        # --- Output norm ---
        if config.use_output_norm:
            self.output_norm = nn.LayerNorm(D)
        else:
            self.output_norm = nn.Identity()

        # Store metadata
        self.n_visual_tokens = N
        self.max_action_tokens = config.max_action_tokens
        self._seq_len = N + 1 + N  # state + action + query = 19

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the copied LLM layers using peft.

        Must be called after __init__. Separated so that the caller can
        control when peft is imported and applied.
        """
        from peft import LoraConfig, get_peft_model

        lora_cfg = self.config.lora
        peft_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
        )

        # Wrap self.layers with LoRA
        # We create a temporary wrapper module so peft can find the target modules
        class _LayersWrapper(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

        wrapper = _LayersWrapper(self.layers)
        wrapper = get_peft_model(wrapper, peft_config)

        # Extract the LoRA-wrapped layers back
        self.layers = wrapper.model.layers

    def _add_embeddings(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add token_type, spatial, and time embeddings."""
        B = state_tokens.shape[0]
        N = self.n_visual_tokens
        device = state_tokens.device

        if self.config.use_token_type_embed:
            state_tokens = state_tokens + self.token_type_embed(
                torch.zeros(N, dtype=torch.long, device=device)
            )
            action_tokens = action_tokens + self.token_type_embed(
                torch.ones(1, dtype=torch.long, device=device)
            )
            query_tokens = query_tokens + self.token_type_embed(
                torch.full((N,), 2, dtype=torch.long, device=device)
            )

        if self.config.use_spatial_embed:
            spatial_ids = torch.arange(N, device=device)
            spatial = self.spatial_embed(spatial_ids)
            state_tokens = state_tokens + spatial
            query_tokens = query_tokens + spatial

        if self.config.use_time_embed:
            current_time = self.time_embed(torch.zeros(1, dtype=torch.long, device=device))
            future_time = self.time_embed(torch.ones(1, dtype=torch.long, device=device))
            state_tokens = state_tokens + current_time
            action_tokens = action_tokens + current_time
            query_tokens = query_tokens + future_time

        return state_tokens, action_tokens, query_tokens

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_tokens:        (B, N, D) current-state visual tokens.
            action_tokens:       (B, 1, D) action embedding (discrete).
            action_padding_mask: (B, A) bool — True = padding.  Optional.

        Returns:
            predicted_tokens: (B, N, D) predicted next-state tokens.
        """
        B, num_state, D = state_tokens.shape
        num_action = action_tokens.shape[1]

        if num_state != self.n_visual_tokens:
            raise ValueError(
                f"Expected {self.n_visual_tokens} state tokens, got {num_state}.",
            )

        # Expand prediction queries
        query_tokens = self.prediction_queries.expand(B, -1, -1)

        # Apply input adapters
        state_tokens = self.state_adapter(state_tokens)
        action_tokens = self.action_adapter(action_tokens)
        query_tokens = self.query_adapter(query_tokens)

        # Add learned embeddings (token_type, spatial, time)
        state_tokens, action_tokens, query_tokens = self._add_embeddings(
            state_tokens, action_tokens, query_tokens,
        )

        # Concatenate: [state(9) | action(1) | query(9)] = 19 tokens
        hidden_states = torch.cat([state_tokens, action_tokens, query_tokens], dim=1)
        seq_len = hidden_states.shape[1]

        # Build position_ids for RoPE: 0..seq_len-1 (preserve native)
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Build attention mask: bidirectional (no causal mask)
        # For Qwen layers, we need a 4D attention mask where 0 = attend, -inf = mask
        # Full bidirectional = all zeros (attend everywhere)
        if action_padding_mask is not None:
            # Build padding mask: True = padding position
            state_mask = torch.zeros(B, num_state, device=hidden_states.device, dtype=torch.bool)
            query_mask = torch.zeros(B, self.n_visual_tokens, device=hidden_states.device, dtype=torch.bool)
            full_padding_mask = torch.cat(
                [state_mask, action_padding_mask.to(hidden_states.device), query_mask],
                dim=1,
            )
            # Convert to 4D attention mask: (B, 1, seq_len, seq_len)
            # padding positions should have -inf in attention
            expanded_mask = full_padding_mask[:, None, None, :]  # (B, 1, 1, seq_len)
            attention_mask = expanded_mask.float() * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(B, 1, seq_len, seq_len)
        else:
            attention_mask = None

        # Forward through all Qwen decoder layers (bidirectional — no causal mask)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            # Qwen decoder layers return a tuple: (hidden_states, ...)
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # Extract query positions
        predicted = hidden_states[:, num_state + num_action:, :]
        return self.output_norm(predicted)


def build_predictor(
    config: JEPAWorldModelConfig,
    vlm_model=None,
) -> nn.Module:
    """Factory function to build the appropriate predictor.

    Args:
        config: World model configuration.
        vlm_model: The Qwen2.5-VL model instance. Required when
            predictor_type == "qwen_lora".

    Returns:
        A predictor module with forward(state_tokens, action_tokens, action_padding_mask).
    """
    if config.predictor_type == "independent_transformer":
        return JEPAPredictor(config)

    if config.predictor_type == "qwen_lora":
        if vlm_model is None:
            raise ValueError(
                "vlm_model is required for predictor_type='qwen_lora'. "
                "Pass the Qwen2.5-VL model so LLM layers can be copied."
            )

        # Extract LLM decoder layers from Qwen2.5-VL
        # Qwen2VLForConditionalGeneration.model.layers is the decoder stack
        if hasattr(vlm_model, "model") and hasattr(vlm_model.model, "layers"):
            llm_layers = vlm_model.model.layers
        elif hasattr(vlm_model, "layers"):
            llm_layers = vlm_model.layers
        else:
            raise AttributeError(
                "Cannot find LLM decoder layers in vlm_model. "
                "Expected vlm_model.model.layers or vlm_model.layers."
            )

        predictor = QwenBidirectionalPredictor(config, llm_layers)
        predictor._apply_lora()
        return predictor

    raise ValueError(f"Unknown predictor_type: {config.predictor_type}")
