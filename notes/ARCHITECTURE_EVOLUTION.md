# Architecture Evolution: SDXL → Clean Baseline → Bounded Attention

## Phase 0: Initial Pipeline (2026-04-26)
- Naive SDXL UNet pipeline with LLM parser + sequential frame generation
- Rule-heavy prompt compiler with 80+ hardcoded verb/noun/entity rules

## Phase 1: SCA + Batch Generation (2026-04-27 to 2026-05-02)
- ConsistentSelfAttentionProcessor: window attention, layer-wise strength, CFG split
- Batch generation: pipe(prompt=list, N-frames) with sequential fallback
- CLIP=0.286 Consist=0.413 Overall=0.324

## Phase 2: Entity Contract (2026-05-07)
- CharacterState + PanelState with identity_core, wardrobe_state, expected_count
- Prompt compiler: 12-slot priority assembly with budget enforcement
- CLIP=0.305 Consist=0.399 Overall=0.343

## Phase 3: Overfitting Removal (Round 1, 2026-05-10)
- Removed 80+ hardcoded rules (all entity-type-specific, verb-specific, prop-specific patterns)
- Established "move upstream to LLM" philosophy
- Overall=0.346 (within -0.001 of rule-heavy baseline, validating overfitting hypothesis)

## Phase 4: Token Merging + Bounded Attention (Rounds 3-5, 2026-05-10)
- Storybooth-style token merging with layer-aware thresholds
- CapturingCrossAttentionProcessor for cross-attention map extraction
- Bounded self-attention masks for multi-character region isolation
- Overall=0.338 (stable, 32/32, 0 frozen)

## Phase 5: Wardrobe Palette Split (Round 6, 2026-05-11)
- _CLOTHING_COLORS / _FACE_COLORS separation in llm_parser.py
- clothing_palette + face_palette dataclass fields
- Unconditional palette merge in prompt compiler
- wardrobe_drift reduced from 3 to 2

## Phase 6: Animal Identity Fix (Current, 2026-05-11)
- Color-token bonus in _select_identity_terms (not just hair)
- Generic term penalty for bare markers like "fur", "coat"
- Per-character nonhuman boost (not per-story)
- Animal feature confusion addressed

## Current Metrics (2026-05-11)
CLIP=0.307  Consist=0.384  Overall=0.338  Pass=32/32  Frozen=0
wardrobe_drift=2  count_drift=1  scene_overwrite=5  rel_prop_loss=5

## Key Principle
"Move knowledge upstream to the LLM" — instead of hardcoding rules in pipeline.py, add generalizable principles to the LLM system prompt that apply to any story.
