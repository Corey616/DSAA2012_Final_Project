# StoryGen Architecture Reference

## Pipeline Flow
Script (.txt) → LLM Parser (Qwen2.5-7B) → ProductionBoard + StoryState → Prompt Compiler (pipeline.py) → SDXL UNet + SCA → Batch Generation → 3 images → CLIP/LPIPS/BLIP-2 Evaluation

## Key Components
| Component | File | Lines |
|-----------|------|-------|
| LLM Parser | llm_parser.py | ~2650 |
| Pipeline | pipeline.py | ~2524 |
| SCA | consistent_self_attn.py | ~475 |
| Cross-Attn Capture | cross_attn_capture.py | ~199 |
| CLIP Eval | metric_clip.py | 152 |
| Consist Eval | metric_consistency.py | 253 |
| Vision Eval | vision_eval.py | 1402 |
| Failure Atlas | failure_atlas.py | 415 |

## SCA Architecture
- ConsistentSelfAttentionProcessor: window attention (±K), layer-wise strength (down=0.14, mid=0.24, up=0.07), step gating (start_ratio=0.35)
- Token merging: Storybooth-style, layer-aware threshold (mid=0.55, up=0.75), step-decay weight
- Bounded attention: CapturingCrossAttentionProcessor → extract_character_masks() → set_bounded_masks()

## Iteration History
| Round | Change | Overall | wardrobe_drift |
|-------|--------|---------|---------------|
| 0 (baseline) | Rule-heavy compiler | 0.348 | 5 |
| 1 | Remove 80+ overfitted rules | 0.346 | 4 |
| 2 | Garment terms + color filter | 0.347 | 4 |
| 3 | SCA penalty + bounded infra | 0.346 | 4 |
| 4 | Token merging + LLM rules | 0.347 | 3 |
| 5 | Bounded attention pipeline | 0.338 | 3 |
| 6 | Wardrobe palette split (THIS LOOP) | 0.338 | 2 |

## Current State
CLIP=0.307  Consist=0.384  Overall=0.338  Pass=32/32  Frozen=0
wardrobe_drift=2  count_drift=1  scene_overwrite=5  rel_prop_loss=5

## Active Improvements
- ✅ Token merging (mid_block=0.55, up=0.75, step-decay)
- ✅ Bounded attention masks (custom cross-attn processors)
- ✅ Clothing/face palette split (clothing_palette + face_palette)
- ✅ Garment term scoring with color filter
