from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def pool_token_sequence(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=1)


def bidirectional_infonce(
    predicted_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, float]:
    logits = predicted_embeddings @ target_embeddings.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    ) / 2
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def token_cosine_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    cosine = F.cosine_similarity(
        predicted_tokens,
        target_tokens.detach(),
        dim=-1,
    )
    return (1.0 - cosine).mean()


def compute_world_model_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    predicted_reward: Optional[torch.Tensor] = None,
    target_reward: Optional[torch.Tensor] = None,
    *,
    lambda_nce: float = 1.0,
    lambda_cos: float = 1.0,
    lambda_r: float = 0.1,
    temperature: float = 0.07,
    proj_head_pred: Optional[torch.nn.Module] = None,
    proj_head_tgt: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = predicted_tokens.new_tensor(0.0)

    cosine_loss = token_cosine_loss(predicted_tokens, target_tokens)
    total_loss = total_loss + lambda_cos * cosine_loss

    infonce_loss = predicted_tokens.new_tensor(0.0)
    infonce_acc = 0.0
    infonce_skipped = 0.0
    if lambda_nce > 0.0 and predicted_tokens.shape[0] > 1:
        num_tokens = predicted_tokens.shape[1]
        token_losses = []
        token_accs = []

        for token_idx in range(num_tokens):
            predicted_token = F.normalize(predicted_tokens[:, token_idx], dim=-1)
            target_token = F.normalize(target_tokens[:, token_idx].detach(), dim=-1)

            if proj_head_pred is not None and proj_head_tgt is not None:
                predicted_token = F.normalize(proj_head_pred(predicted_token), dim=-1)
                target_token = F.normalize(proj_head_tgt(target_token), dim=-1)

            token_loss, token_acc = bidirectional_infonce(
                predicted_embeddings=predicted_token,
                target_embeddings=target_token,
                temperature=temperature,
            )
            token_losses.append(token_loss)
            token_accs.append(token_acc)

        infonce_loss = torch.stack(token_losses).mean()
        infonce_acc = float(sum(token_accs) / len(token_accs))
        total_loss = total_loss + lambda_nce * infonce_loss
    else:
        infonce_skipped = 1.0

    reward_loss = predicted_tokens.new_tensor(0.0)
    if predicted_reward is not None and target_reward is not None:
        reward_loss = F.mse_loss(
            predicted_reward.float(), target_reward.detach().float(),
        ).to(predicted_tokens.dtype)
        total_loss = total_loss + lambda_r * reward_loss

    metrics = {
        "loss/total": total_loss.item(),
        "loss/infonce": infonce_loss.item(),
        "loss/cosine": cosine_loss.item(),
        "loss/reward": reward_loss.item(),
        "metric/infonce_acc": infonce_acc,
        "metric/cosine_sim": 1.0 - cosine_loss.item(),
        "metric/infonce_skipped": infonce_skipped,
    }
    return total_loss, metrics


def multi_horizon_loss(
    step_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    horizon_weight_decay: float = 0.9,
    normalize_weights: bool = False,
    **loss_kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not step_results:
        raise ValueError("step_results must not be empty.")

    device = step_results[0][0].device
    weights = torch.tensor(
        [horizon_weight_decay ** step for step in range(len(step_results))],
        device=device,
        dtype=step_results[0][0].dtype,
    )
    if normalize_weights:
        weights = weights / weights.sum()

    total_loss = torch.zeros((), device=device, dtype=weights.dtype)
    metrics: Dict[str, float] = {}

    for step, (pred_tokens, tgt_tokens, pred_reward, tgt_reward) in enumerate(step_results):
        step_loss, step_metrics = compute_world_model_loss(
            pred_tokens,
            tgt_tokens,
            pred_reward,
            tgt_reward,
            **loss_kwargs,
        )
        total_loss = total_loss + weights[step] * step_loss
        for key, value in step_metrics.items():
            metrics[f"{key}/step_{step}"] = value
        metrics[f"metric/step_weight/step_{step}"] = float(weights[step].item())

    metrics["loss/total_multihorizon"] = total_loss.item()
    return total_loss, metrics
