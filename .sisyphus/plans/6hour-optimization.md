# StoryGen 6-Hour Optimization Plan

## Context
- Backbone: SD3.5-Medium (StableDiffusion3Pipeline)
- LLM: Qwen3-4B-Instruct-2507
- 8× A800 80GB, multi-GPU parallel works
- SCA disabled, IP-Adapter not integrated
- Rule-based fallbacks stripped

## Current Visual Issues (post SD3.5 migration)
1. Penetration/clipping artifacts on characters
2. Detail inconsistency (clothing patterns, textures)
3. Action/scene capture not precise enough
4. Cross-frame consistency is GOOD overall
5. open_clip evaluation broken (corrupted cache)

## Tier 1 — PARALLEL (Hours 0-2)

### T1: Img2Img Frame Harmonization
- File: `src/core_generator/pipeline.py`
- Lines: 518-595 (frame loop), 553-563 (call_kwargs), 73 (import)
- Change: For frame i≥1, use `StableDiffusion3Img2ImgPipeline.from_pipe(base_pipe)` with previous frame as init_image, strength=0.65-0.70
- QA: Generate 2-frame story, verify LPIPS ≤ 0.45

### T2: Prompt Identity Anchoring
- File: `src/core_generator/pipeline.py`
- Lines: 166-327 (_compose_prompt), 536-545 (negative_prompt)
- Change: Inject "same person, identical face, same exact clothing" into prompts; add identity-drift negative prompt terms
- QA: grep -c "identical" on generated prompts ≥ 1 per frame

### T4: Color Harmonization
- File: `src/utils/image_utils.py` (new function) + `src/core_generator/pipeline.py` (lines 570-573)
- Change: Add `match_histogram()` for inter-frame color matching with adaptive blend factor
- QA: Frame 3 saturation ≥ Frame 1 saturation

### T3: LLM Prompt Refinement (LOWER PRIORITY)
- File: `src/script_director/llm_parser.py`
- Lines: 71-205 (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE)
- Change: Require 150+ char visual_description, add non-human entity instructions

## Tier 2 (Hours 2-4)

### T5: IP-Adapter FaceID
- Files: `src/core_generator/pipeline.py`, `src/asset_anchor/character_portrait.py`
- Requires: Testing SDXL IP-Adapter compatibility with SD3.5

### T6: SCA Diagnostic
- New file: `scripts/test_sca_diagnostic.py`
- Tests 2-frame generation with SCA enabled vs disabled

## Tier 3 (Hours 4-6)

### T7: Evaluation Fix
- File: `src/evaluation_hub/metric_clip.py`
- Fix open_clip model loading

### T8: Anime Style Option
- Research switching to animation style to mask texture/pattern inconsistencies
