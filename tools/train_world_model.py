"""Training script for JEPA world model with 3-phase curriculum.

Phase 1: Single-step, teacher-forced (horizon=1)
  → advance when EMA cosine_sim >= threshold AND EMA infonce_acc >= threshold
Phase 2: Multi-step, open-loop detached (horizon=2)
  → advance when EMA cosine_sim >= threshold
Phase 3: Multi-step, open-loop BPTT (horizon=4)

Usage:
    python tools/train_world_model.py \
        --dataset_dir data/sokoban_wm_v1_train5k_val1k \
        --output_dir outputs/qwen_lora_curriculum \
        --vlm_name Qwen/Qwen2.5-VL-3B-Instruct \
        --predictor_type qwen_lora \
        --wandb_project jepa-mbrl
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SokobanEpisodeDataset(Dataset):
    """Single-transition dataset: (s_t, a_t, r_t, s_{t+1})."""

    def __init__(self, episode_dir: str):
        self.episode_dir = Path(episode_dir)
        self.files = sorted(self.episode_dir.glob("episode_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No episode files found in {episode_dir}")
        self.index = []
        for file_idx, f in enumerate(self.files):
            data = np.load(f)
            n = len(data["actions_model"])
            for step_idx in range(n):
                self.index.append((file_idx, step_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, step_idx = self.index[idx]
        data = np.load(self.files[file_idx])
        return {
            "frame_t": data["frames"][step_idx],
            "frame_tp1": data["frames"][step_idx + 1],
            "action": int(data["actions_model"][step_idx]),
            "reward": float(data["rewards"][step_idx]),
        }


class SequenceDataset(Dataset):
    """Multi-step window dataset for multi-horizon training."""

    def __init__(self, episode_dir: str, window: int = 4):
        self.episode_dir = Path(episode_dir)
        self.files = sorted(self.episode_dir.glob("episode_*.npz"))
        self.window = window
        self.index = []
        for file_idx, f in enumerate(self.files):
            data = np.load(f)
            n = len(data["actions_model"])
            if n >= window:
                for start in range(n - window + 1):
                    self.index.append((file_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        data = np.load(self.files[file_idx])
        w = self.window
        return {
            "frames": data["frames"][start : start + w + 1],  # (w+1, H, W, 3)
            "actions": data["actions_model"][start : start + w].astype(np.int64),
            "rewards": data["rewards"][start : start + w].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_frames_batch(visual_encoder, frames_np, processor, device, dtype):
    """Encode numpy frames (B, H, W, 3) → (B, N, D) visual tokens."""
    images = [Image.fromarray(f) for f in frames_np]
    image_inputs = processor.image_processor(images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = image_inputs["image_grid_thw"].to(device=device)
    return visual_encoder.encode_preprocessed(pixel_values, image_grid_thw)


class EMATracker:
    """Exponential moving average for smooth metric tracking."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.values = {}

    def update(self, key: str, value: float):
        if key not in self.values:
            self.values[key] = value
        else:
            self.values[key] = (1 - self.alpha) * self.values[key] + self.alpha * value

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


PHASE_CONFIG = {
    1: {"horizon": 1, "mode": "teacher_forced",      "label": "Phase1:single-step"},
    2: {"horizon": 2, "mode": "open_loop_detached",   "label": "Phase2:horizon-2"},
    3: {"horizon": 4, "mode": "open_loop_bptt",       "label": "Phase3:horizon-4"},
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    # --- wandb ---
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode=args.wandb_mode,
        dir=args.output_dir,
    )

    # --- Load VLM ---
    print(f"Loading VLM: {args.vlm_name}")
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VLMClass

    vlm_model = VLMClass.from_pretrained(args.vlm_name, torch_dtype=dtype, device_map=device)
    vlm_model.eval()
    for p in vlm_model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(args.vlm_name)

    # --- Visual encoder ---
    from vagen.world_model.encoders import FrozenVisualEncoder
    visual_module = getattr(vlm_model, "visual", None) or vlm_model.model.visual
    visual_encoder = FrozenVisualEncoder(visual_module, processor)

    # --- World model ---
    from vagen.world_model.config import JEPAWorldModelConfig, LoRAConfig
    from vagen.world_model.world_model import JEPAWorldModel
    from vagen.world_model.losses import compute_world_model_loss, multi_horizon_loss

    lora_config = LoRAConfig(
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    wm_config = JEPAWorldModelConfig(
        hidden_dim=2048,
        n_visual_tokens=9,
        predictor_type=args.predictor_type,
        n_llm_layers=args.n_llm_layers,
        attention_mode=args.attention_mode,
        n_predictor_layers=args.n_predictor_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        lora=lora_config,
        use_input_adapters=args.use_input_adapters,
        use_token_type_embed=args.use_token_type_embed,
        use_spatial_embed=args.use_spatial_embed,
        use_time_embed=args.use_time_embed,
        action_type="discrete",
        num_actions=4,
        infonce_temperature=args.temperature,
        infonce_loss_weight=args.infonce_weight,
        token_cosine_loss_weight=args.cosine_weight,
        reward_loss_weight=args.reward_weight,
    )

    print(f"Building world model with predictor_type={args.predictor_type}")
    world_model = JEPAWorldModel(
        config=wm_config,
        vlm_model=vlm_model if args.predictor_type == "qwen_lora" else None,
    ).to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in world_model.parameters())
    trainable_params = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable_params:,}")
    wandb.log({"params/total": total_params, "params/trainable": trainable_params})

    # Recurrence norm for multi-step rollout
    recurrence_norm = nn.LayerNorm(wm_config.hidden_dim).to(device=device, dtype=dtype)

    # --- Optimizer ---
    all_trainable = list(world_model.parameters()) + list(recurrence_norm.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in all_trainable if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        adjusted = step - args.warmup_steps
        T = args.lr_restart_period  # steps per cosine cycle
        return 0.5 * (1.0 + np.cos(np.pi * (adjusted % T) / T))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # --- Datasets ---
    train_dir = os.path.join(args.dataset_dir, "train", "episodes")
    val_dir = os.path.join(args.dataset_dir, "val", "episodes")

    # Single-step dataset (Phase 1)
    single_dataset = SokobanEpisodeDataset(train_dir)
    single_loader = DataLoader(
        single_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Multi-step datasets (Phase 2 & 3)
    seq2_dataset = SequenceDataset(train_dir, window=2)
    seq2_loader = DataLoader(
        seq2_dataset, batch_size=args.multistep_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    seq4_dataset = SequenceDataset(train_dir, window=4)
    seq4_loader = DataLoader(
        seq4_dataset, batch_size=args.multistep_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Val loader (always single-step for consistent eval)
    val_dataset = SokobanEpisodeDataset(val_dir)
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # --- Curriculum state ---
    current_phase = 1
    ema = EMATracker(alpha=args.ema_alpha)
    single_iter = iter(single_loader)
    seq2_iter = iter(seq2_loader)
    seq4_iter = iter(seq4_loader)

    def get_single_batch():
        nonlocal single_iter
        try:
            return next(single_iter)
        except StopIteration:
            single_iter = iter(single_loader)
            return next(single_iter)

    def get_seq_batch(horizon):
        nonlocal seq2_iter, seq4_iter
        if horizon == 2:
            try:
                return next(seq2_iter)
            except StopIteration:
                seq2_iter = iter(seq2_loader)
                return next(seq2_iter)
        else:
            try:
                return next(seq4_iter)
            except StopIteration:
                seq4_iter = iter(seq4_loader)
                return next(seq4_iter)

    # --- Training loop ---
    print(f"Starting 3-phase curriculum training for {args.max_train_steps} steps")
    print(f"Phase 1→2: cos>={args.phase1_cos_threshold}, nce>={args.phase1_nce_threshold}")
    print(f"Phase 2→3: cos>={args.phase2_cos_threshold}")
    world_model.train()

    for step in range(1, args.max_train_steps + 1):
        phase_cfg = PHASE_CONFIG[current_phase]
        horizon = phase_cfg["horizon"]
        mode = phase_cfg["mode"]

        if current_phase == 1:
            # --- Phase 1: Single-step ---
            batch = get_single_batch()
            with torch.no_grad():
                state_tokens = encode_frames_batch(
                    visual_encoder, batch["frame_t"].numpy(), processor, device, dtype,
                )
                target_tokens = encode_frames_batch(
                    visual_encoder, batch["frame_tp1"].numpy(), processor, device, dtype,
                )
            actions = batch["action"].to(device)
            rewards = batch["reward"].to(device, dtype=dtype)

            _, _, loss, metrics = world_model(
                state_tokens, actions, target_tokens, rewards,
            )
        else:
            # --- Phase 2 & 3: Multi-step ---
            batch = get_seq_batch(horizon)
            frames_np = batch["frames"].numpy()  # (B, H+1, 96, 96, 3)
            B = frames_np.shape[0]

            # Encode all frames
            with torch.no_grad():
                all_frames = frames_np.reshape(-1, *frames_np.shape[2:])  # (B*(H+1), 96, 96, 3)
                all_tokens = encode_frames_batch(
                    visual_encoder, all_frames, processor, device, dtype,
                )  # (B*(H+1), N, D)
                N, D = all_tokens.shape[1], all_tokens.shape[2]
                all_tokens = all_tokens.view(B, horizon + 1, N, D)

            actions_all = batch["actions"].to(device)   # (B, H)
            rewards_all = batch["rewards"].to(device, dtype=dtype)  # (B, H)

            # Multi-step rollout
            step_results = []
            current_state = all_tokens[:, 0]  # (B, N, D)

            for h in range(horizon):
                step_actions = actions_all[:, h]
                pred_tokens, pred_reward = world_model.predict(current_state, step_actions)
                tgt_tokens = all_tokens[:, h + 1]
                tgt_reward = rewards_all[:, h]
                step_results.append((pred_tokens, tgt_tokens, pred_reward, tgt_reward))

                # Next state depends on mode
                if mode == "teacher_forced":
                    current_state = tgt_tokens.detach()
                elif mode == "open_loop_detached":
                    current_state = recurrence_norm(pred_tokens.detach())
                elif mode == "open_loop_bptt":
                    current_state = recurrence_norm(pred_tokens)

            loss, metrics = multi_horizon_loss(
                step_results,
                horizon_weight_decay=args.horizon_weight_decay,
                normalize_weights=False,
                lambda_nce=args.infonce_weight,
                lambda_cos=args.cosine_weight,
                lambda_r=args.reward_weight,
                temperature=args.temperature,
                proj_head_pred=world_model.prediction_projection,
                proj_head_tgt=world_model.target_projection,
            )

            # Use step_0 metrics for curriculum tracking
            step0_cos = metrics.get("metric/cosine_sim/step_0", 0.0)
            step0_nce = metrics.get("metric/infonce_acc/step_0", 0.0)
            metrics["metric/cosine_sim"] = step0_cos
            metrics["metric/infonce_acc"] = step0_nce

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad],
                args.max_grad_norm,
            )
        optimizer.step()
        scheduler.step()

        # Update EMA
        cos_sim = metrics.get("metric/cosine_sim", 0.0)
        nce_acc = metrics.get("metric/infonce_acc", 0.0)
        ema.update("cosine_sim", cos_sim)
        ema.update("infonce_acc", nce_acc)

        # Curriculum advancement
        old_phase = current_phase
        if current_phase == 1:
            if (ema.get("cosine_sim") >= args.phase1_cos_threshold
                    and ema.get("infonce_acc") >= args.phase1_nce_threshold):
                current_phase = 2
        elif current_phase == 2:
            if ema.get("cosine_sim") >= args.phase2_cos_threshold:
                current_phase = 3

        if current_phase != old_phase:
            new_cfg = PHASE_CONFIG[current_phase]
            print(f"\n{'='*60}")
            print(f"  PHASE ADVANCED: {old_phase} → {current_phase}  ({new_cfg['label']})")
            print(f"  EMA cos={ema.get('cosine_sim'):.4f}  nce={ema.get('infonce_acc'):.4f}")
            print(f"  horizon={new_cfg['horizon']}, mode={new_cfg['mode']}")
            print(f"{'='*60}\n")
            wandb.log({
                "curriculum/phase_change": current_phase,
                "curriculum/phase_change_step": step,
            }, step=step)

        # Log gate statistics
        predictor = world_model.predictor
        if hasattr(predictor, "_last_gate_mean"):
            metrics["gate/mean"] = predictor._last_gate_mean
            for i, g in enumerate(getattr(predictor, "_last_gate_per_token", [])):
                metrics[f"gate/token_{i}"] = g

        # Log
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["curriculum/phase"] = current_phase
        metrics["curriculum/horizon"] = horizon
        metrics["ema/cosine_sim"] = ema.get("cosine_sim")
        metrics["ema/infonce_acc"] = ema.get("infonce_acc")

        if step % 10 == 0:
            wandb.log(metrics, step=step)
            phase_label = PHASE_CONFIG[current_phase]["label"]
            print(
                f"[Step {step}/{args.max_train_steps}] [{phase_label}] "
                f"loss={metrics.get('loss/total', metrics.get('loss/total_multihorizon', 0)):.4f} "
                f"cos={cos_sim:.4f}(ema:{ema.get('cosine_sim'):.4f}) "
                f"nce={nce_acc:.4f}(ema:{ema.get('infonce_acc'):.4f}) "
                f"lr={metrics['lr']:.2e}"
            )

        # Eval
        if step % args.eval_every == 0:
            val_metrics = evaluate(
                world_model, visual_encoder, val_loader, processor,
                device, dtype, args.val_max_batches,
            )
            val_log = {f"val/{k}": v for k, v in val_metrics.items()}
            val_log["val/curriculum_phase"] = current_phase
            wandb.log(val_log, step=step)
            print(
                f"  [Val] loss={val_metrics['loss/total']:.4f} "
                f"cos={val_metrics['metric/cosine_sim']:.4f} "
                f"nce={val_metrics['metric/infonce_acc']:.4f}"
            )

        # Save
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            save_checkpoint(world_model, recurrence_norm, optimizer, scheduler, step, current_phase, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "checkpoints", "final.pt")
    save_checkpoint(world_model, recurrence_norm, optimizer, scheduler, args.max_train_steps, current_phase, final_path)
    print(f"Training complete. Final: {final_path}  Phase reached: {current_phase}")
    wandb.finish()


@torch.no_grad()
def evaluate(world_model, visual_encoder, val_loader, processor, device, dtype, max_batches):
    world_model.eval()
    all_metrics = {}
    count = 0
    for batch in val_loader:
        if count >= max_batches:
            break
        state_tokens = encode_frames_batch(
            visual_encoder, batch["frame_t"].numpy(), processor, device, dtype,
        )
        target_tokens = encode_frames_batch(
            visual_encoder, batch["frame_tp1"].numpy(), processor, device, dtype,
        )
        actions = batch["action"].to(device)
        rewards = batch["reward"].to(device, dtype=dtype)
        _, _, _, metrics = world_model(state_tokens, actions, target_tokens, rewards)
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        count += 1
    world_model.train()
    if count > 0:
        for k in all_metrics:
            all_metrics[k] /= count
    return all_metrics


def save_checkpoint(world_model, recurrence_norm, optimizer, scheduler, step, phase, path):
    trainable_state = {
        k: v for k, v in world_model.state_dict().items()
        if any(p.data_ptr() == v.data_ptr()
               for p in world_model.parameters() if p.requires_grad)
    }
    torch.save({
        "step": step,
        "phase": phase,
        "model_state_dict": trainable_state,
        "recurrence_norm_state_dict": recurrence_norm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train JEPA world model (3-phase curriculum)")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_lora_curriculum")

    # Model
    parser.add_argument("--vlm_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--predictor_type", type=str, default="qwen_lora",
                        choices=["qwen_lora", "independent_transformer"])
    parser.add_argument("--n_llm_layers", type=int, default=36)
    parser.add_argument("--attention_mode", type=str, default="bidirectional")
    parser.add_argument("--n_predictor_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--ffn_dim", type=int, default=5504)
    parser.add_argument("--dropout", type=float, default=0.0)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])

    # Adapters & embeddings
    parser.add_argument("--use_input_adapters", action="store_true", default=True)
    parser.add_argument("--no_input_adapters", dest="use_input_adapters", action="store_false")
    parser.add_argument("--use_token_type_embed", action="store_true", default=True)
    parser.add_argument("--no_token_type_embed", dest="use_token_type_embed", action="store_false")
    parser.add_argument("--use_spatial_embed", action="store_true", default=True)
    parser.add_argument("--no_spatial_embed", dest="use_spatial_embed", action="store_false")
    parser.add_argument("--use_time_embed", action="store_true", default=True)
    parser.add_argument("--no_time_embed", dest="use_time_embed", action="store_false")

    # Training
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--multistep_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_restart_period", type=int, default=1000,
                        help="Steps per cosine restart cycle")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_max_batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Loss
    parser.add_argument("--infonce_weight", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--reward_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--horizon_weight_decay", type=float, default=0.9)

    # Curriculum thresholds
    parser.add_argument("--phase1_cos_threshold", type=float, default=0.81,
                        help="EMA cosine_sim threshold to advance Phase1→2")
    parser.add_argument("--phase1_nce_threshold", type=float, default=0.70,
                        help="EMA infonce_acc threshold to advance Phase1→2")
    parser.add_argument("--phase2_cos_threshold", type=float, default=0.80,
                        help="EMA cosine_sim threshold to advance Phase2→3")
    parser.add_argument("--ema_alpha", type=float, default=0.05,
                        help="EMA smoothing factor for curriculum metrics")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="jepa-mbrl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    args = parser.parse_args()
    if args.wandb_run_name is None:
        args.wandb_run_name = f"sokoban-curriculum-{args.predictor_type}-{int(time.time())}"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
