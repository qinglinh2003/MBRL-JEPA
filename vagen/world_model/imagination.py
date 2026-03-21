from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from .config import ImaginationConfig, TrajectoryMode
from .encoders import FrozenVisualEncoder
from .world_model import JEPAWorldModel


@dataclass
class ImaginedStep:
    predicted_tokens: torch.Tensor
    predicted_reward: torch.Tensor


@dataclass
class ImaginedTrajectory:
    steps: List[ImaginedStep] = field(default_factory=list)

    @property
    def rewards(self) -> torch.Tensor:
        if not self.steps:
            return torch.empty(0)
        return torch.stack([step.predicted_reward for step in self.steps], dim=1)

    @property
    def states(self) -> torch.Tensor:
        if not self.steps:
            return torch.empty(0)
        return torch.stack([step.predicted_tokens for step in self.steps], dim=1)

    @property
    def horizon(self) -> int:
        return len(self.steps)

    def discounted_return(self, gamma: float) -> torch.Tensor:
        rewards = self.rewards
        if rewards.numel() == 0:
            return torch.empty(0)
        returns = torch.zeros_like(rewards[:, 0])
        for step in reversed(range(rewards.shape[1])):
            returns = rewards[:, step] + gamma * returns
        return returns


class LatentImagination(nn.Module):
    """Pure latent rollout core plus optional bridge back into a VLM actor."""

    def __init__(
        self,
        visual_encoder: FrozenVisualEncoder,
        world_model: JEPAWorldModel,
        config: Optional[ImaginationConfig] = None,
        *,
        vlm=None,
        processor=None,
        prompt_builder: Optional[Callable[[str], list]] = None,
        image_token_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.world_model = world_model
        self.config = config or ImaginationConfig()
        self.recurrence_norm = nn.LayerNorm(world_model.config.hidden_dim)

        self.vlm = vlm
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.image_token_id = image_token_id
        self.device = device or next(world_model.parameters()).device
        self._dummy_image = Image.new("RGB", self.config.image_size, color=(0, 0, 0))

    @torch.no_grad()
    def encode_observation(self, images: Sequence[Image.Image] | Image.Image) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        return self.visual_encoder.encode_images(images).float()

    @torch.no_grad()
    def imagine(
        self,
        state_tokens: torch.Tensor,
        actions: Any,
        horizon: Optional[int] = None,
        mode: TrajectoryMode = "open_loop_detached",
    ) -> ImaginedTrajectory:
        if mode == "teacher_forced":
            raise ValueError("teacher_forced requires target states and is only supported in rollout_for_training().")

        if state_tokens.ndim == 2:
            current_state = state_tokens.unsqueeze(0).to(self.device)
        else:
            current_state = state_tokens.to(self.device)
        batch_size = int(current_state.shape[0])

        effective_horizon = self._infer_horizon(actions, batch_size)
        if horizon is not None:
            effective_horizon = min(effective_horizon, horizon)
        effective_horizon = min(effective_horizon, self.config.max_horizon)

        trajectory = ImaginedTrajectory()
        for step in range(effective_horizon):
            step_actions = self._step_actions(actions, step, batch_size)
            predicted_tokens, predicted_reward = self.world_model.predict(
                current_state,
                step_actions,
            )
            trajectory.steps.append(
                ImaginedStep(
                    predicted_tokens=predicted_tokens,
                    predicted_reward=predicted_reward,
                ),
            )

            if mode == "open_loop_detached":
                current_state = self.recurrence_norm(predicted_tokens.detach())
            elif mode == "open_loop_bptt":
                current_state = self.recurrence_norm(predicted_tokens)
            else:
                raise ValueError(f"Unsupported imagination mode: {mode}")

        return trajectory

    @torch.no_grad()
    def imagine_from_images(
        self,
        images: Sequence[Image.Image] | Image.Image,
        actions: Any,
        horizon: Optional[int] = None,
        mode: TrajectoryMode = "open_loop_detached",
    ) -> ImaginedTrajectory:
        state_tokens = self.encode_observation(images)
        return self.imagine(
            state_tokens=state_tokens,
            actions=actions,
            horizon=horizon,
            mode=mode,
        )

    def rollout_for_training(
        self,
        initial_state: torch.Tensor,
        actions: Any,
        target_states: torch.Tensor,
        target_rewards: torch.Tensor,
        horizon: int,
        mode: TrajectoryMode,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        results = []
        current_state = initial_state
        batch_size = int(initial_state.shape[0])

        for step in range(horizon):
            step_actions = self._step_actions(actions, step, batch_size)
            predicted_tokens, predicted_reward = self.world_model.predict(
                current_state,
                step_actions,
            )
            target_tokens = target_states[:, step]
            target_reward = target_rewards[:, step]
            results.append(
                (predicted_tokens, target_tokens, predicted_reward, target_reward),
            )

            if mode == "teacher_forced":
                current_state = target_tokens.detach()
            elif mode == "open_loop_detached":
                current_state = self.recurrence_norm(predicted_tokens.detach())
            elif mode == "open_loop_bptt":
                current_state = self.recurrence_norm(predicted_tokens)
            else:
                raise ValueError(f"Unsupported training mode: {mode}")

        return results

    @torch.no_grad()
    def inject_into_vlm(
        self,
        predicted_tokens: torch.Tensor,
        action_text: str,
    ) -> torch.Tensor:
        """Optional policy bridge for reusing a VLM actor on imagined states."""
        if self.vlm is None or self.processor is None or self.prompt_builder is None:
            raise ValueError(
                "VLM bridge requires vlm, processor, and prompt_builder to be set.",
            )
        if self.image_token_id is None:
            self.image_token_id = self.vlm.config.image_token_id

        if predicted_tokens.ndim != 2:
            raise ValueError(
                "inject_into_vlm expects a single imagined state shaped (N, D).",
            )

        messages = self.prompt_builder(action_text)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            images=[self._dummy_image],
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_thw = inputs["image_grid_thw"]
        inputs_embeds = self.vlm.model.embed_tokens(input_ids)

        image_mask = input_ids == self.image_token_id
        num_image_tokens = int(image_mask.sum().item())
        if num_image_tokens != predicted_tokens.shape[0]:
            raise ValueError(
                f"Expected {num_image_tokens} predicted tokens, got {predicted_tokens.shape[0]}.",
            )

        mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        predicted_tokens = predicted_tokens.to(
            device=self.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = inputs_embeds.masked_scatter(mask_3d, predicted_tokens)

        outputs = self.vlm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _infer_horizon(self, actions: Any, batch_size: int) -> int:
        if self.world_model.config.action_type == "discrete":
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions)
            if actions.ndim == 0:
                return 1
            if actions.ndim == 1:
                return int(actions.shape[0]) if batch_size == 1 else 1
            return int(actions.shape[1])

        if len(actions) == 0:
            return 0
        if isinstance(actions[0], str):
            return len(actions) if batch_size == 1 else 1
        return len(actions[0])

    def _step_actions(self, actions: Any, step: int, batch_size: int) -> Any:
        if self.world_model.config.action_type == "discrete":
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions, device=self.device)
            if actions.ndim == 0:
                if step > 0:
                    raise IndexError("Single discrete action only supports step=0.")
                return actions.unsqueeze(0)
            if actions.ndim == 1:
                if batch_size == 1 and actions.shape[0] > 1:
                    return actions[step : step + 1]
                if step > 0:
                    raise IndexError("Single-step discrete action batch only supports step=0.")
                return actions
            return actions[:, step]

        if isinstance(actions[0], str):
            if batch_size == 1 and len(actions) > 1:
                return [actions[step]]
            if step > 0:
                raise IndexError("Single-step text action batch only supports step=0.")
            return actions
        return [sample_actions[step] for sample_actions in actions]
