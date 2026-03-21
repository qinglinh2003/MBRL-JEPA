# JEPA-MBRL

JEPA-style Model-Based Reinforcement Learning for Vision-Language Models.

Core code extracted from the VAGEN framework, focused on latent world model + MBRL.

## Structure

```text
implementations/
├── GPT/                  # GPT-side JEPA world model implementation
├── Claude/               # Claude-side JEPA world model implementation
vagen/
├── env/
│   ├── base/             # Abstract env/service interfaces
│   ├── sokoban/          # Primary test environment
│   └── utils/            # Parsing, reward, state matching
├── trainer/              # PPO/GRPO training (Ray-based)
├── rollout/              # Rollout managers (base + Qwen + inference)
├── inference/            # Model inference interfaces (vLLM, OpenAI, Claude, etc.)
├── server/               # HTTP env server, serialization, LLM-as-judge
└── utils/                # Seeds, scoring
```

## GPT Implementation Files

```text
implementations/GPT/
├── __init__.py
├── config.py
├── encoders.py
├── predictor.py
├── heads.py
├── losses.py
├── world_model.py
├── imagination.py
├── rl_integration.py
└── trainer_mixin.py
```

## Install

```bash
pip install -e .
```

## Implementation Tracking

- GPT-side implementation notes live in `implementations/GPT/`
- Design documents live in `docs/`

## Notes

- GPT and Claude world model implementations are currently isolated under
  `implementations/` for side-by-side comparison before merge.
- Only Sokoban environment is included. Other envs can be added as needed.
- Removed from original repo: docs, public assets, benchmark/example scripts, mkdocs config.
