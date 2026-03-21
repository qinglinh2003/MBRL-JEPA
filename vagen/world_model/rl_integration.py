import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import CurriculumConfig, ImaginationConfig, TrajectoryMode
from .imagination import ImaginedTrajectory, LatentImagination
from .losses import multi_horizon_loss
from .world_model import JEPAWorldModel


@dataclass
class WorldModelTransition:
    state_tokens: torch.Tensor
    action: Any
    reward: float
    next_state_tokens: torch.Tensor
    done: bool


@dataclass
class WorldModelManagerConfig:
    batch_size: int = 64
    multistep_batch_size: int = 32
    buffer_capacity: int = 100_000
    discount: float = 0.99
    rollout_mode: TrajectoryMode = "open_loop_detached"


class WorldModelReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data: deque[WorldModelTransition] = deque(maxlen=capacity)

    def add(self, transition: WorldModelTransition) -> None:
        self.data.append(
            WorldModelTransition(
                state_tokens=transition.state_tokens.detach().cpu(),
                action=transition.action,
                reward=float(transition.reward),
                next_state_tokens=transition.next_state_tokens.detach().cpu(),
                done=bool(transition.done),
            ),
        )

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> List[WorldModelTransition]:
        sample_size = min(batch_size, len(self.data))
        return random.sample(list(self.data), sample_size)


class SequenceBuffer:
    """Stores episode-local transitions for multi-step sampling."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.episodes: deque[list[WorldModelTransition]] = deque()
        self.current_episode: list[WorldModelTransition] = []
        self.transition_count = 0

    def add(self, transition: WorldModelTransition) -> None:
        cpu_transition = WorldModelTransition(
            state_tokens=transition.state_tokens.detach().cpu(),
            action=transition.action,
            reward=float(transition.reward),
            next_state_tokens=transition.next_state_tokens.detach().cpu(),
            done=bool(transition.done),
        )
        self.current_episode.append(cpu_transition)
        self.transition_count += 1

        if cpu_transition.done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while self.transition_count > self.capacity and self.episodes:
            dropped = self.episodes.popleft()
            self.transition_count -= len(dropped)

    def __len__(self) -> int:
        return self.transition_count

    def sample_windows(
        self,
        batch_size: int,
        window: int,
        device: torch.device,
        action_type: str,
    ) -> Optional[Dict[str, Any]]:
        valid_starts: list[tuple[int, int]] = []
        episodes = list(self.episodes)
        for episode_idx, episode in enumerate(episodes):
            if len(episode) < window:
                continue
            for start_idx in range(len(episode) - window + 1):
                valid_starts.append((episode_idx, start_idx))

        if not valid_starts:
            return None

        sample_size = min(batch_size, len(valid_starts))
        sampled = random.sample(valid_starts, sample_size)

        initial_states = []
        target_states = []
        rewards = []
        if action_type == "discrete":
            actions = []
        else:
            actions = []

        for episode_idx, start_idx in sampled:
            window_transitions = episodes[episode_idx][start_idx : start_idx + window]
            initial_states.append(window_transitions[0].state_tokens)
            target_states.append(
                torch.stack([transition.next_state_tokens for transition in window_transitions]),
            )
            rewards.append(
                torch.tensor(
                    [transition.reward for transition in window_transitions],
                    dtype=torch.float32,
                ),
            )

            if action_type == "discrete":
                actions.append(
                    torch.tensor(
                        [transition.action for transition in window_transitions],
                        dtype=torch.long,
                    ),
                )
            else:
                actions.append([transition.action for transition in window_transitions])

        batch: Dict[str, Any] = {
            "initial_state": torch.stack(initial_states).to(device),
            "target_states": torch.stack(target_states).to(device),
            "rewards": torch.stack(rewards).to(device),
        }
        if action_type == "discrete":
            batch["actions"] = torch.stack(actions).to(device)
        else:
            batch["actions"] = actions
        return batch


_PHASE_MODES = {
    1: "teacher_forced",
    2: "open_loop_detached",
    3: "open_loop_bptt",
}


class WorldModelManager:
    """Training lifecycle, replay buffers, curriculum, and imagination."""

    def __init__(
        self,
        world_model: JEPAWorldModel,
        optimizer: torch.optim.Optimizer,
        imagination: Optional[LatentImagination] = None,
        *,
        visual_encoder=None,
        config: Optional[WorldModelManagerConfig] = None,
        imagination_config: Optional[ImaginationConfig] = None,
        curriculum_config: Optional[CurriculumConfig] = None,
        device: Optional[str] = None,
    ):
        self.world_model = world_model
        self.optimizer = optimizer
        self.config = config or WorldModelManagerConfig()
        self.curriculum_config = curriculum_config or world_model.config.curriculum
        self.device = device or str(next(world_model.parameters()).device)

        if imagination is not None:
            self.imagination = imagination
        else:
            if visual_encoder is None:
                raise ValueError(
                    "Either imagination or visual_encoder must be provided.",
                )
            self.imagination = LatentImagination(
                visual_encoder=visual_encoder,
                world_model=world_model,
                config=imagination_config or ImaginationConfig(),
                device=self.device,
            )

        self.replay_buffer = WorldModelReplayBuffer(self.config.buffer_capacity)
        self.sequence_buffer = SequenceBuffer(self.config.buffer_capacity)

        self.current_phase = 1
        self.current_horizon = 1
        self.train_steps = 0

    def collect_transition(
        self,
        state_tokens: torch.Tensor,
        action: Any,
        reward: float,
        next_state_tokens: torch.Tensor,
        done: bool,
    ) -> None:
        transition = WorldModelTransition(
            state_tokens=state_tokens.detach().cpu(),
            action=action,
            reward=float(reward),
            next_state_tokens=next_state_tokens.detach().cpu(),
            done=bool(done),
        )
        self.replay_buffer.add(transition)
        self.sequence_buffer.add(transition)

    def collect_transitions(
        self,
        transitions: Sequence[WorldModelTransition],
    ) -> None:
        for transition in transitions:
            self.collect_transition(
                state_tokens=transition.state_tokens,
                action=transition.action,
                reward=transition.reward,
                next_state_tokens=transition.next_state_tokens,
                done=transition.done,
            )

    def _collate_single_step(
        self,
        batch: Sequence[WorldModelTransition],
    ) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        state_tokens = torch.stack(
            [transition.state_tokens for transition in batch],
            dim=0,
        ).to(self.device)
        next_state_tokens = torch.stack(
            [transition.next_state_tokens for transition in batch],
            dim=0,
        ).to(self.device)
        rewards = torch.tensor(
            [transition.reward for transition in batch],
            device=self.device,
            dtype=torch.float32,
        )

        if self.world_model.config.action_type == "discrete":
            actions = torch.tensor(
                [transition.action for transition in batch],
                device=self.device,
            )
        else:
            actions = [transition.action for transition in batch]
        return state_tokens, actions, rewards, next_state_tokens

    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        if len(self.replay_buffer) == 0:
            return {}

        batch = self.replay_buffer.sample(batch_size or self.config.batch_size)
        state_tokens, actions, rewards, next_state_tokens = self._collate_single_step(batch)

        _, _, loss, metrics = self.world_model(
            state_tokens,
            actions,
            next_state_tokens,
            rewards,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        self._maybe_advance_phase(metrics)
        metrics["curriculum/phase"] = float(self.current_phase)
        metrics["curriculum/horizon"] = float(self.current_horizon)
        metrics["buffer/replay_size"] = float(len(self.replay_buffer))
        return metrics

    def train_step_multistep(
        self,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        horizon = self.current_horizon
        mode = _PHASE_MODES[self.current_phase]
        device = torch.device(self.device)

        batch = self.sequence_buffer.sample_windows(
            batch_size=batch_size or self.config.multistep_batch_size,
            window=horizon,
            device=device,
            action_type=self.world_model.config.action_type,
        )
        if batch is None:
            return {}

        step_results = self.imagination.rollout_for_training(
            initial_state=batch["initial_state"],
            actions=batch["actions"],
            target_states=batch["target_states"],
            target_rewards=batch["rewards"],
            horizon=horizon,
            mode=mode,
        )

        loss, metrics = multi_horizon_loss(
            step_results,
            horizon_weight_decay=self.curriculum_config.horizon_weight_decay,
            normalize_weights=self.curriculum_config.normalize_horizon_weights,
            lambda_nce=self.world_model.config.infonce_loss_weight,
            lambda_cos=self.world_model.config.token_cosine_loss_weight,
            lambda_r=self.world_model.config.reward_loss_weight,
            temperature=self.world_model.config.infonce_temperature,
            proj_head_pred=self.world_model.prediction_projection,
            proj_head_tgt=self.world_model.target_projection,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        step0_metrics = {
            key.replace("/step_0", ""): value
            for key, value in metrics.items()
            if key.endswith("/step_0")
        }
        self._maybe_advance_phase(step0_metrics)
        metrics["curriculum/phase"] = float(self.current_phase)
        metrics["curriculum/horizon"] = float(self.current_horizon)
        metrics["curriculum/mode"] = mode
        metrics["buffer/sequence_size"] = float(len(self.sequence_buffer))
        return metrics

    def imagine_batch(
        self,
        state_tokens: torch.Tensor,
        action_sequences: Any,
        horizon: Optional[int] = None,
        mode: Optional[TrajectoryMode] = None,
    ) -> ImaginedTrajectory:
        rollout_mode = self.config.rollout_mode if mode is None else mode
        was_training = self.world_model.training
        self.world_model.eval()
        try:
            return self.imagination.imagine(
                state_tokens=state_tokens,
                actions=action_sequences,
                horizon=horizon,
                mode=rollout_mode,
            )
        finally:
            self.world_model.train(was_training)

    @torch.no_grad()
    def compute_imagined_returns(
        self,
        trajectory: ImaginedTrajectory,
        discount: Optional[float] = None,
    ) -> torch.Tensor:
        gamma = self.config.discount if discount is None else discount
        return trajectory.discounted_return(gamma)

    def _maybe_advance_phase(self, metrics: Dict[str, float]) -> None:
        cfg = self.curriculum_config
        if self.current_phase == 1:
            cos_ok = metrics.get("metric/cosine_sim", 0.0) >= cfg.phase1_cosine_threshold
            nce_ok = metrics.get("metric/infonce_acc", 0.0) >= cfg.phase1_infonce_acc_threshold
            if cos_ok and nce_ok:
                self.current_phase = 2
                self.current_horizon = cfg.max_horizon_phase2
        elif self.current_phase == 2:
            cos_ok = metrics.get("metric/cosine_sim", 0.0) >= cfg.phase2_cosine_threshold
            if cos_ok:
                self.current_phase = 3
                self.current_horizon = cfg.max_horizon_phase3
