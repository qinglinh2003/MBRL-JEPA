from typing import Any, Optional

import torch
import torch.nn as nn

from .config import JEPAWorldModelConfig
from .encoders import ActionEncoding, DiscreteActionEncoder, TextActionEncoder
from .heads import ProjectionHead, RewardHead
from .losses import compute_world_model_loss, pool_token_sequence
from .predictor import build_predictor


class JEPAWorldModel(nn.Module):
    """Single-step JEPA world model operating on visual token states."""

    def __init__(
        self,
        config: JEPAWorldModelConfig,
        tokenizer=None,
        token_embedding: Optional[nn.Module] = None,
        vlm_model=None,
    ):
        """
        Args:
            config: World model configuration.
            tokenizer: Required for text action encoding.
            token_embedding: Optional shared token embedding from VLM.
            vlm_model: The Qwen2.5-VL model instance. Required when
                config.predictor_type == "qwen_lora" so that LLM layers
                can be deep-copied into the predictor.
        """
        super().__init__()
        self.config = config

        # Build predictor (independent transformer or Qwen+LoRA)
        self.predictor = build_predictor(config, vlm_model=vlm_model)

        self.reward_head = RewardHead(config.hidden_dim)

        if config.action_type == "discrete":
            self.action_encoder = DiscreteActionEncoder(
                num_actions=config.num_actions,
                hidden_dim=config.hidden_dim,
            )
        elif config.action_type == "text":
            if tokenizer is None:
                raise ValueError("tokenizer is required for text action encoding.")
            self.action_encoder = TextActionEncoder(
                tokenizer=tokenizer,
                token_embedding=token_embedding,
                hidden_dim=config.hidden_dim,
                vocab_size=config.action_vocab_size,
                max_action_tokens=config.max_action_tokens,
                freeze_token_embedding=config.freeze_text_token_embedding,
            )
        else:
            raise ValueError(f"Unsupported action_type: {config.action_type}")

        self.prediction_projection = None
        self.target_projection = None
        if config.projection_head.enabled:
            self.prediction_projection = ProjectionHead(
                input_dim=config.hidden_dim,
                output_dim=config.projection_head.shared_dim,
                use_mlp=config.projection_head.use_mlp,
            )
            self.target_projection = ProjectionHead(
                input_dim=config.hidden_dim,
                output_dim=config.projection_head.shared_dim,
                use_mlp=config.projection_head.use_mlp,
            )

    def _encode_action(self, actions: Any) -> ActionEncoding:
        return self.action_encoder(actions)

    def predict(
        self,
        state_tokens: torch.Tensor,
        actions: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_encoding = self._encode_action(actions)
        predicted_tokens = self.predictor(
            state_tokens=state_tokens,
            action_tokens=action_encoding.tokens.to(state_tokens.device),
            action_padding_mask=action_encoding.padding_mask,
        )
        pooled_prediction = pool_token_sequence(predicted_tokens)
        predicted_reward = self.reward_head(pooled_prediction)
        return predicted_tokens, predicted_reward

    def predict_next_tokens(
        self,
        state_tokens: torch.Tensor,
        actions: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict(state_tokens, actions)

    def compute_loss(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        target_reward: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_world_model_loss(
            predicted_tokens,
            target_tokens,
            predicted_reward,
            target_reward,
            lambda_nce=self.config.infonce_loss_weight,
            lambda_cos=self.config.token_cosine_loss_weight,
            lambda_r=self.config.reward_loss_weight,
            temperature=self.config.infonce_temperature,
            proj_head_pred=self.prediction_projection,
            proj_head_tgt=self.target_projection,
        )

    def forward(
        self,
        state_tokens: torch.Tensor,
        actions: Any,
        target_tokens: torch.Tensor,
        target_reward: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        predicted_tokens, predicted_reward = self.predict(state_tokens, actions)
        loss, metrics = self.compute_loss(
            predicted_tokens,
            target_tokens,
            predicted_reward,
            target_reward,
        )
        return predicted_tokens, predicted_reward, loss, metrics
