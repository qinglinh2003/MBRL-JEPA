from dataclasses import dataclass, field
from typing import Literal, List


TrajectoryMode = Literal[
    "teacher_forced",
    "open_loop_detached",
    "open_loop_bptt",
]


@dataclass
class ProjectionHeadConfig:
    enabled: bool = False
    shared_dim: int = 1536
    use_mlp: bool = False


@dataclass
class CurriculumConfig:
    phase1_cosine_threshold: float = 0.95
    phase1_infonce_acc_threshold: float = 0.80
    phase2_cosine_threshold: float = 0.90
    max_horizon_phase2: int = 2
    max_horizon_phase3: int = 4
    horizon_weight_decay: float = 0.9
    normalize_horizon_weights: bool = False


@dataclass
class ImaginationConfig:
    max_horizon: int = 4
    discount: float = 0.99
    image_size: tuple[int, int] = (96, 96)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    unfreeze_norms: bool = True


@dataclass
class JEPAWorldModelConfig:
    hidden_dim: int = 2048
    n_visual_tokens: int = 9

    # Predictor backbone
    predictor_type: Literal["independent_transformer", "qwen_lora"] = "qwen_lora"
    n_llm_layers: int = 36
    attention_mode: Literal["bidirectional", "causal", "prefix_lm"] = "bidirectional"

    # Independent transformer settings (only used when predictor_type == "independent_transformer")
    n_predictor_layers: int = 4
    n_heads: int = 16
    ffn_dim: int = 5504
    dropout: float = 0.0
    predictor_init: Literal["random", "llm_layers"] = "random"
    use_learned_positional_embeddings: bool = True

    # LoRA (only used when predictor_type == "qwen_lora")
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Input adapters
    use_input_adapters: bool = True
    adapter_type: Literal["rmsnorm_linear_residual", "bottleneck"] = "rmsnorm_linear_residual"
    adapter_bottleneck_dim: int = 256

    # Learned embeddings
    use_token_type_embed: bool = True
    use_spatial_embed: bool = True
    use_time_embed: bool = True

    # Position encoding
    rope_mode: Literal["preserve_native"] = "preserve_native"

    # Output
    use_output_norm: bool = True
    use_recurrence_norm: bool = True

    # Action
    action_type: Literal["discrete", "text"] = "discrete"
    num_actions: int = 4
    max_action_tokens: int = 16
    action_vocab_size: int = 151936
    freeze_text_token_embedding: bool = True

    # Loss
    infonce_temperature: float = 0.07
    infonce_loss_weight: float = 1.0
    token_cosine_loss_weight: float = 1.0
    reward_loss_weight: float = 0.1

    projection_head: ProjectionHeadConfig = field(
        default_factory=ProjectionHeadConfig,
    )
    curriculum: CurriculumConfig = field(
        default_factory=CurriculumConfig,
    )
