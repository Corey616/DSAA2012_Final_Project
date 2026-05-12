---
name: storygen-code
description: >-
  Maps all storygen package code locations and dependencies. Covers SCA key
  principles (Q unchanged, K/V concat), pipeline.py frame generation loop,
  consistent_self_attn.py processor architecture, and evaluation metrics.
  When modifying attention or generation code, use this skill first to
  understand existing structure and avoid breaking dependencies.
license: MIT
compatibility:
  - python >= 3.10
  - torch >= 2.0
metadata:
  priority: high
  last_updated: 2026-05-02
---

# StoryGen Code Map

## Module Architecture
```
storygen/
├── script_director/
│   ├── llm_parser.py          # ProductionBoard, LLMScriptParser (1110 lines)
│   ├── llm_parser_local.py    # LocalQwenParser (1317 lines)
│   └── prompt_enhancer.py     # PromptEnhancer (210 lines)
├── core_generator/
│   ├── pipeline.py            # NarrativeGenerationPipeline (747 lines)
│   ├── memory_bank.py         # MemoryBank (193 lines)
│   └── attention/
│       └── consistent_self_attn.py  # 3 processors (318 lines)
├── asset_anchor/
│   └── character_portrait.py  # CharacterPortraitGenerator (238 lines)
├── evaluation_hub/
│   ├── metric_clip.py         # CLIPEvaluator (152 lines)
│   └── metric_consistency.py  # ConsistencyEvaluator (253 lines)
├── orchestrator/
│   └── run_pipeline.py        # CLI entry point (204 lines)
└── utils/
    ├── image_utils.py         # Image processing (272 lines)
    ├── mirror_config.py       # HF mirror setup (224 lines)
    └── text_parser.py         # Script parsing (106 lines)
```

## SCA Key Principles
1. **Q must NOT be modified** — K/V alone provide cross-frame context
2. **Batch generation required** — sequential for-loop cannot share attention
3. **Dimension matching** — VAE latent (4-dim) ≠ UNet attention dims; needs projection
4. **Layer-wise strength** — shallow layers (scene structure): high strength; deep layers (details): low strength

## Key Dependencies
- `pipeline.py` imports from: `character_portrait.py`, `memory_bank.py`, `consistent_self_attn.py`
- `run_pipeline.py` imports from: `llm_parser.py`, `prompt_enhancer.py`, `pipeline.py`
- `run_taska_batch.py` imports from: `llm_parser_local.py`, `pipeline.py`, `metric_clip.py`, `metric_consistency.py`
