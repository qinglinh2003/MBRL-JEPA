from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class ActionEncoding:
    tokens: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def split_visual_tokens_by_image(
    merged_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Split flattened merged tokens into a `(B, N, D)` tensor."""
    tokens_per_image = []
    for thw in image_grid_thw:
        t, h, w = thw.tolist()
        tokens_per_image.append(
            int(t * (h // spatial_merge_size) * (w // spatial_merge_size)),
        )

    if len(set(tokens_per_image)) != 1:
        raise ValueError(
            "Variable numbers of visual tokens per image are not supported in the "
            "current GPT merged implementation.",
        )

    token_chunks = merged_tokens.split(tokens_per_image, dim=0)
    return torch.stack(token_chunks, dim=0)


class FrozenVisualEncoder(nn.Module):
    """Thin wrapper around a frozen Qwen-style visual encoder."""

    def __init__(self, visual_encoder: nn.Module, image_processor=None):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.image_processor = image_processor
        self.spatial_merge_size = getattr(visual_encoder, "spatial_merge_size", 2)
        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)
        self.visual_encoder.eval()

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(
            device=_module_device(self.visual_encoder),
            dtype=_module_dtype(self.visual_encoder),
        )
        image_grid_thw = image_grid_thw.to(device=_module_device(self.visual_encoder))
        return self.visual_encoder(pixel_values, grid_thw=image_grid_thw)

    @torch.no_grad()
    def encode_preprocessed(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        merged_tokens = self.forward(pixel_values, image_grid_thw)
        return split_visual_tokens_by_image(
            merged_tokens=merged_tokens,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )

    @torch.no_grad()
    def encode_images(self, images: Sequence) -> torch.Tensor:
        if self.image_processor is None:
            raise ValueError("image_processor is required for encode_images().")
        image_inputs = self.image_processor(images, return_tensors="pt")
        return self.encode_preprocessed(
            pixel_values=image_inputs["pixel_values"],
            image_grid_thw=image_inputs["image_grid_thw"],
        )

    @torch.no_grad()
    def encode(self, images: Sequence) -> torch.Tensor:
        return self.encode_images(images)

    @property
    def hidden_dim(self) -> int:
        if hasattr(self.visual_encoder, "merger") and hasattr(self.visual_encoder.merger, "mlp"):
            last_layer = self.visual_encoder.merger.mlp[-1]
            if hasattr(last_layer, "out_features"):
                return int(last_layer.out_features)
        raise AttributeError("Could not infer hidden_dim from visual encoder.")

    def train(self, mode: bool = True):
        return super().train(False)

    @classmethod
    def from_vlm(cls, vlm_model, processor) -> "FrozenVisualEncoder":
        return cls(vlm_model.visual, processor)


class DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.embedding = nn.Embedding(num_actions, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, actions: torch.Tensor) -> ActionEncoding:
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)
        actions = actions.long()
        if actions.numel() > 0:
            min_action = int(actions.min().item())
            max_action = int(actions.max().item())
            if min_action < 0 or max_action >= self.num_actions:
                raise ValueError(
                    "Discrete actions must be remapped into "
                    f"[0, {self.num_actions - 1}] before entering the world model. "
                    f"Got action range [{min_action}, {max_action}].",
                )
        tokens = self.embedding(actions).unsqueeze(1)
        padding_mask = torch.zeros(
            tokens.shape[:2],
            device=tokens.device,
            dtype=torch.bool,
        )
        return ActionEncoding(tokens=tokens, padding_mask=padding_mask)


class TextActionEncoder(nn.Module):
    def __init__(
        self,
        tokenizer,
        hidden_dim: int,
        token_embedding: Optional[nn.Module] = None,
        vocab_size: Optional[int] = None,
        max_action_tokens: int = 16,
        freeze_token_embedding: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_action_tokens = max_action_tokens

        if token_embedding is None:
            if vocab_size is None:
                vocab_size = len(tokenizer)
            token_embedding = nn.Embedding(vocab_size, hidden_dim)
            freeze_token_embedding = False

        self.token_embedding = token_embedding
        if freeze_token_embedding:
            for param in self.token_embedding.parameters():
                param.requires_grad_(False)

        embed_dim = getattr(self.token_embedding, "embedding_dim", hidden_dim)
        self.projection = (
            nn.Identity()
            if embed_dim == hidden_dim
            else nn.Linear(embed_dim, hidden_dim)
        )

    def forward(self, actions: Sequence[str]) -> ActionEncoding:
        batch = self.tokenizer(
            list(actions),
            padding=True,
            truncation=True,
            max_length=self.max_action_tokens,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        device = _module_device(self.token_embedding)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        token_embeddings = self.token_embedding(input_ids)
        token_embeddings = self.projection(token_embeddings)
        padding_mask = attention_mask == 0
        return ActionEncoding(tokens=token_embeddings, padding_mask=padding_mask)
