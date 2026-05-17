# StoryGen Next Optimization Plan — 6-Hour Sprint

**Date**: 2026-05-17
**Backbone**: SD3.5-Medium + Qwen3-4B-Instruct-2507
**Style**: Anime/Ghibli (clean lineart, cel shading, flat colors)
**Key Constraint**: NO img2img, NO IP-Adapter without explicit testing gate
**Goal**: Fix style inconsistency, action/scene mismatch, and underutilized negative prompts

---

## Pre-Flight Findings (from codebase exploration)

### Critical Bugs Found
1. **`prompt_enhancer.py` blocks anime in its negative prompt** (lines 138-148: blocks "cartoon", "anime style", "illustration") — directly contradicts the anime goal
2. **`prompt_enhancer.py` STYLE_MODIFIERS are all photorealistic** (lines 30-56: "photorealistic photography, cinematic lighting") — outputs never used but still confusing
3. **`character_portrait.py` uses photorealistic portraits** (line 90: "cinematic portrait photography") — mismatched with anime frame style
4. **`llm_parser.py` LLM is instructed to generate per-panel negatives** (STEP 6B, lines 197-203) but `parse_llm_response()` NEVER extracts them — half-implemented feature
5. **SD3.5 triple-encoder negative prompts untapped** — `negative_prompt_2` (CLIP-G) and `negative_prompt_3` (T5) never used
6. **`consistency_constraints` field flows through entire pipeline but is never consumed** at generation time

### Architecture Gaps
- Style string repeated in 3+ places with no single source of truth
- No style anchoring — style is a suffix, not weighted at prompt start
- SD3.5 guidance_scale at 7.5 (recommended for SD3.5 Medium: 5.0)
- Negative prompts are 20+ terms (SDXL-era pattern) — SD3.5 works best with 5-10 terms

---

## TIER 1: Quick Wins & Bug Fixes (Hours 0–1.5)

### T1.1 — Fix `prompt_enhancer.py` Style Conflict ⚠️ CRITICAL
**File**: `storygen/src/script_director/prompt_enhancer.py`
**Lines**: 30-56 (STYLE_MODIFIERS), 109-154 (create_negative_prompt)

**Changes**:
- Replace `STYLE_MODIFIERS` photorealistic entries with anime/Ghibli variants:
  ```
  "anime_ghibli": "anime style, studio ghibli, soft watercolor tones, gentle lighting, hand-drawn look"
  "anime_cinematic": "anime style, cinematic composition, dramatic shading, cel-style highlights"
  "anime_slice_of_life": "anime style, warm slice-of-life tones, clean lineart, flat shading"
  ```
- Remove `cartoon, anime style, illustration, anime, manga, 2D art style` from `create_negative_prompt()`
- Add anime-specific negatives: `semi-realistic, photorealistic, 3D render, CGI, Western comic style, inconsistent style`
- Remove the `"BLOCKED"` annotation on `whimsical_illustration`

**QA**:
```bash
python -c "from src.script_director.prompt_enhancer import PromptEnhancer; p = PromptEnhancer(); print(p.create_negative_prompt())"
# Expected: Does NOT contain "anime", "cartoon", "manga", "illustration"
# Expected: DOES contain "semi-realistic", "photorealistic", "Western comic style"
```

### T1.2 — Wire Up LLM-Generated Per-Panel Negative Prompts
**File**: `storygen/src/script_director/llm_parser.py`
**Lines**: 645-754 (`parse_llm_response()` panel building loop)

**Changes**:
- At line ~670 (inside the panel_data extraction block), add:
  ```python
  panel_negative = panel_data.get('negative_prompt', '') if panel_data and isinstance(panel_data, dict) else ''
  ```
- Store it in `panel_data_with_defaults` dict (line 740):
  ```python
  'negative_prompt': panel_negative,
  ```
- The `Panel` dataclass already has `negative_prompt: str = ""` at line 43 — just needs population

**File**: `storygen/src/core_generator/pipeline.py`
**Lines**: 535-558 (`generate_story()` negative prompt section)

**Changes**:
- Replace hardcoded negative_prompt with:
  ```python
  # Use LLM-generated per-panel negative if available, fall back to base negative
  base_negative = (
      "blurry, deformed, bad anatomy, extra limbs, missing limbs, "
      "watermark, text, signature, low quality, jpeg artifacts, "
      "semi-realistic, photorealistic, 3D render, CGI, Western comic style, "
      "different art style, inconsistent style, mixed media"
  )
  panel_negative = getattr(panel, 'negative_prompt', '')
  negative_prompt = panel_negative if panel_negative else base_negative
  ```

**QA**:
```bash
python -c "
from src.script_director.llm_parser_local import create_qwen_parser
parser = create_qwen_parser()
with parser:
    board = parser.process_script_file('data/TaskA/01.txt')
    # Verify panels have negative_prompt populated
    for p in board.panels:
        print(f'Panel {p.panel_id}: neg_prompt={p.negative_prompt[:80]}...')
"
# Expected: Non-empty negative_prompt for each panel, containing style-specific blockers
```

### T1.3 — Enable SD3.5 Triple Negative Prompt
**File**: `storygen/src/core_generator/pipeline.py`
**Lines**: 550-559 (`call_kwargs` in `generate_story()`)

**Changes**:
```python
call_kwargs = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "negative_prompt_2": negative_prompt,  # CLIP-G: same negatives, different encoding
    "negative_prompt_3": negative_prompt,  # T5-XXL: nuanced negative (longer context window)
    "height": height,
    "width": width,
    "num_inference_steps": gen_params.get("num_steps", 35),
    "guidance_scale": gen_params.get("guidance_scale", 5.0),  # SD3.5 Medium optimal
    "generator": generator,
}
```
- For advanced use, `negative_prompt_3` (T5) can carry **longer, more nuanced** negatives since T5 has 256-token context vs CLIP's 77-token limit.

**QA**:
```bash
# Verify the pipeline accepts the new kwargs without errors
python -c "
from diffusers import StableDiffusion3Pipeline
# Check that __call__ signature includes negative_prompt_2, negative_prompt_3
import inspect
sig = inspect.signature(StableDiffusion3Pipeline.__call__)
print([p for p in sig.parameters if 'negative' in p])
# Expected: ['negative_prompt', 'negative_prompt_2', 'negative_prompt_3']
"
```

### T1.4 — Fix `character_portrait.py` Anime Style
**File**: `storygen/src/asset_anchor/character_portrait.py`
**Lines**: 90-99 (portrait prompt and negative prompt)

**Changes**:
- Line 90-95: Change portrait prompt from `"cinematic portrait photography"` → `"anime style portrait, studio ghibli, clean lineart, cel shading, soft shading, hand-drawn look"`
- Line 97-99: Change negative prompt from blocking anime → blocking photorealistic: `"photorealistic, realistic photography, 3D render, CGI, Western comic style"`

**QA**:
```python
# grep for "photorealistic" in character_portrait.py - should only appear in NEGATIVE prompt
grep -n "photorealistic" storygen/src/asset_anchor/character_portrait.py
```

### T1.5 — Fix SD3.5 Guidance Scale
**File**: `storygen/scripts/run_worker.py` (line 71), `storygen/scripts/run_taska_batch.py` (line 58)
**Change**: `"guidance_scale": 7.5` → `"guidance_scale": 5.0` (per Stability AI recommendations for SD3.5 Medium)

**QA**: No code verification needed — numeric value change only. Benchmark after full batch run.

---

## TIER 2: Style Anchoring & Prompt Quality (Hours 1.5–3.5)

### T2.1 — Style Anchor Injection (WEIGHTED at prompt START)
**File**: `storygen/src/core_generator/pipeline.py`
**Lines**: 283-327 (`_compose_prompt()` Step 3-4)

**Problem**: Style is currently a suffix. SD3.5 (MMDiT) attends to prompt holistically, but earlier tokens get slightly more conditioning from the CLIP encoders.

**Changes**:
- Move style from end to BEGINNING of prompt (after character, before scene):
  ```python
  # NEW prompt order:
  # [style_anchor] → [character_desc] → [scene_action] → [key_objects] → [time] → [setting]
  
  STYLE_ANCHOR = "anime style, studio ghibli, clean lineart, cel shading, flat colors, hand-drawn look"
  
  prompt_parts = []
  
  # 0. Style anchor FIRST for stronger conditioning
  prompt_parts.append(STYLE_ANCHOR)
  
  # 1. Character description
  if combined_char:
      prompt_parts.append(combined_char)
  
  # ... rest unchanged
  ```
- Remove style from line 307 (it's now at position 0)
- Update truncation logic (lines 315-326) to protect the style anchor

**QA**:
```bash
python -c "
from src.core_generator.pipeline import NarrativeGenerationPipeline
# Instantiate pipeline, call _compose_prompt on a test panel
# Verify prompt starts with 'anime style, studio ghibli...'
"
# Expected: prompt.startswith("anime style, studio ghibli, clean lineart")
```

### T2.2 — Multi-Scale Style Injection
**File**: `storygen/src/core_generator/pipeline.py`
**Lines**: 283-327 (`_compose_prompt()`)

**Rationale**: SD3.5 has 3 text encoders with different capacities. We can inject style at multiple positions for redundancy.

**Changes**:
- Style anchor at START (T2.1 above) — picked up by all 3 encoders
- Style suffix at END — `"consistent art style across all frames"` — for T5's longer context
- Per-panel style tag injected into the LLM's `enhanced_prompt` field via SYSTEM_PROMPT instruction

```python
# At the end of prompt composition:
if not base_prompt.endswith("consistent art style"):
    base_prompt += ", consistent art style across all frames"
```

**QA**:
```bash
python -c "
# Verify every generated prompt ends with 'consistent art style across all frames'
"
```

### T2.3 — Strengthen LLM Negative Prompt Instruction
**File**: `storygen/src/script_director/llm_parser.py`
**Lines**: 197-204 (STEP 6B in USER_PROMPT_TEMPLATE)

**Current instruction is good but add:**
- MUST produce `negative_prompt` for EACH panel (make non-optional)
- Add anime-specific negative prompt guidance:
  ```
  ### STEP 6B: Negative Prompts (negative_prompts) — MANDATORY
  For EACH panel, generate a negative_prompt that blocks:
  - Style drift: "semi-realistic, photorealistic, 3D render, CGI, Western comic style, 
    different art style, inconsistent style, mixed media, realistic texture, realistic lighting"
  - Anatomy: "blurry face, poorly drawn face, asymmetrical eyes, bad anatomy"
  - Scene contamination: "different location, missing key objects, extra person"
  - Quality: "low quality, watermark, text, signature, jpeg artifacts"
  - Scene-specific: If in KITCHEN → "outdoor, street, office"; if in PARK → "indoor, room"
  ```

### T2.4 — Unify Style Strings to Single Source of Truth
**File**: NEW — `storygen/src/script_director/style_config.py`
**Change**: Create a centralized style config module:

```python
# style_config.py — Single source of truth for all style strings
ANIME_GHIBLI_POSITIVE = (
    "anime style, studio ghibli, clean lineart, cel shading, "
    "flat colors, soft shading, hand-drawn look"
)

ANIME_GHIBLI_NEGATIVE = (
    "semi-realistic, photorealistic, 3D render, CGI, Western comic style, "
    "different art style, inconsistent style, mixed media, realistic texture, "
    "realistic lighting, detailed texture patterns"
)

STYLE_CONSISTENCY_LOCK = "consistent art style across all frames"

STYLE_OPTIONS = {
    "anime_ghibli": {
        "positive": ANIME_GHIBLI_POSITIVE,
        "negative": ANIME_GHIBLI_NEGATIVE,
        "consistency": STYLE_CONSISTENCY_LOCK,
    },
    "anime_cinematic": {
        "positive": "anime style, cinematic anime, dramatic lighting, cel-shaded, detailed backgrounds",
        "negative": ANIME_GHIBLI_NEGATIVE + ", flat shading, simplistic background, watercolor",
        "consistency": STYLE_CONSISTENCY_LOCK,
    },
}
```

Then import from this module in pipeline.py, llm_parser.py, prompt_enhancer.py, and character_portrait.py.

**QA**:
```bash
# Verify no hardcoded style strings remain in pipeline.py, prompt_enhancer.py
grep -n "anime style" storygen/src/core_generator/pipeline.py
grep -n "anime style" storygen/src/script_director/prompt_enhancer.py
# Expected: imports from style_config, not inline strings
```

---

## TIER 3: AB Testing & Evaluation (Hours 3.5–6.0)

### T3.1 — AB Testing Framework
**File**: NEW — `storygen/scripts/run_ab_test.py`

**Purpose**: Compare baseline (current) vs optimized (new changes) on identical seeds.

```python
#!/usr/bin/env python3
"""
AB Testing Framework
Compares baseline vs candidate on identical seeds across all 32 cases.
Outputs: per-case side-by-side storyboards, metric deltas, statistical summary
"""
# Usage:
# python run_ab_test.py --baseline outputs/baseline --candidate outputs/candidate --cases all
# Output: outputs/ab_test/storyboard_diff_01.png, metrics_comparison.json, summary.md
```

**Implementation**:
1. Run all 32 cases with `seed=42` using the BASELINE code (git stash changes first)
2. Apply optimization changes
3. Run all 32 cases with `seed=42` using OPTIMIZED code
4. For each case, produce:
   - Side-by-side storyboard (baseline top row, candidate bottom row)
   - Per-frame CLIP score comparison
   - Per-story LPIPS consistency comparison
   - Subjective style consistency rating (anime coherence score via CLIP similarity to Ghibli reference images)

**QA**:
```bash
python run_ab_test.py --baseline outputs/ab_baseline --candidate outputs/ab_candidate --cases 01 06 20
# Expected: Creates comparison storyboards and JSON metrics diff
```

### T3.2 — Automated Style Consistency Metric
**File**: NEW — `storygen/src/evaluation_hub/metric_style_consistency.py`

**Implementation**: Use a CLIP-based "anime coherence" scorer:
```python
class StyleConsistencyEvaluator:
    """
    Measures how consistently anime/Ghibli the generated frames are.
    Computes cosine similarity between each frame's CLIP embedding
    and a set of Ghibli reference image embeddings.
    """
    GHIBLI_REFERENCE_PROMPTS = [
        "a Studio Ghibli anime frame, hand-drawn animation, clean lineart, cel shading",
        "anime style, studio ghibli background art, soft watercolor, gentle lighting",
    ]
    
    def compute_style_score(self, image: Image, clip_model) -> float:
        """Returns 0-1 score of how Ghibli-like the image is"""
        ...
    
    def compute_consistency(self, images: List[Image]) -> float:
        """Returns variance of style scores across frames (lower = more consistent)"""
        ...
```

**QA**:
```bash
python -c "
from src.evaluation_hub.metric_style_consistency import StyleConsistencyEvaluator
evaluator = StyleConsistencyEvaluator(device='cuda:0')
# Test on existing outputs
score = evaluator.compute_style_score(Image.open('outputs/taskA_batch/01/frame_01.png'))
print(f'Style score: {score:.3f}')
# Expected: ~0.7-0.9 for anime outputs, ~0.2-0.4 for photorealistic
"
```

### T3.3 — Full 32-Case Benchmark Run
**Script**: `storygen/scripts/run_parallel.py`

**Command**:
```bash
python storygen/scripts/run_parallel.py \
  --max-gpus 8 \
  --output outputs/optimization_v1 \
  --min-free-gb 30
```

**Target Metrics** (vs current baseline):
| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| Avg CLIP Score | ~0.30 | ≥0.33 | Better action/scene alignment from improved prompts |
| Avg LPIPS Consistency | ~0.30 | ≥0.35 | Better style anchoring reduces drift |
| Style variance (new) | N/A | ≤0.10 | New metric: std dev of anime coherence across frames |
| Generation time | 8.9 min | ≤9.5 min | Acceptable overhead from triple negative encoding |

**QA**:
```bash
python storygen/scripts/run_parallel.py --max-gpus 8 --output outputs/optimization_v1
# After completion:
python -c "
import json
with open('outputs/optimization_v1/parallel_summary.json') as f:
    data = json.load(f)
    success = sum(1 for r in data['results'] if r.get('status')=='success')
    print(f'Success: {success}/{data[\"total_cases\"]}')
"
# Expected: 32/32 success, metrics meet targets above
```

### T3.4 — Failure Analysis & Iteration
**Time**: 30 min
**Script**: Use `failure-atlas` skill on optimization_v1 outputs

```bash
# Identify weak cases
python -c "
import json
from pathlib import Path

summary = json.load(open('outputs/optimization_v1/parallel_summary.json'))
for r in summary['results']:
    if r.get('status') == 'success':
        metrics = json.loads(r.get('metrics', '{}'))
        if isinstance(metrics, dict):
            clip = metrics.get('avg_clip_score', 0)
            cons = metrics.get('avg_consistency', 0)
            if clip < 0.25 or cons < 0.25:
                print(f'WEAK: {Path(r[\"script\"]).stem} CLIP={clip:.3f} CONS={cons:.3f}')
"
# Apply last-mile prompt tweaks to the weakest 3 cases
```

---

## Design Principles (MUST NOT Violate)

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **NO img2img** | Binds scene structure, prevents scene changes across frames |
| 2 | **NO IP-Adapter without testing gate** | Can freeze face/identity, similar side effects to img2img |
| 3 | **Prompt-level changes first** | Lower risk, faster iteration, easier to roll back |
| 4 | **SD3.5-specific tuning** | guidance_scale=5.0 (not 7.5), short negatives (not 20-term lists) |
| 5 | **NO SCA re-enablement** | Dimension handling issues documented; out of scope |
| 6 | **Keep color harmonization** | T4 histogram matching is working and should remain |

---

## File Change Summary

| File | Tier | Change |
|------|------|--------|
| `prompt_enhancer.py` | T1.1 | Fix style conflict (anime modifiers, remove anime from negatives) |
| `llm_parser.py` | T1.2, T2.3 | Extract LLM negative_prompt; strengthen instruction |
| `pipeline.py` | T1.2, T1.3, T2.1, T2.2 | Wire LLM negatives; pass neg2/neg3; style anchor at start |
| `character_portrait.py` | T1.4 | Anime portrait style |
| `run_worker.py` | T1.5 | guidance_scale 7.5→5.0 |
| `run_taska_batch.py` | T1.5 | guidance_scale 7.5→5.0 |
| NEW: `style_config.py` | T2.4 | Single source of truth for all style strings |
| NEW: `run_ab_test.py` | T3.1 | AB testing framework |
| NEW: `metric_style_consistency.py` | T3.2 | Anime coherence scoring |

---

## Rollback Strategy

All changes are in `pipeline.py`, `prompt_enhancer.py`, `llm_parser.py`, `character_portrait.py`, and new files. To rollback:

```bash
git stash                    # Revert pipeline/prompt changes
# New files (style_config.py, run_ab_test.py) are additive and safe to leave
# Batch runner config changes (guidance_scale) are trivial numeric reverts
```

Critical to preserve:
- histogram matching (pipeline.py lines 572-581)
- `_compose_prompt` character extraction logic (lines 184-281)
- `_extract_characters_from_panel` (lines 418-455)

---

## Success Criteria (Agent-Executable)

```bash
# 1. Style conflict resolved
python -c "from src.script_director.prompt_enhancer import PromptEnhancer; n = PromptEnhancer().create_negative_prompt(); assert 'anime' not in n.lower(), 'BUG: anime still blocked'"

# 2. LLM negative prompts wired
python -c "
from src.script_director.llm_parser_local import create_qwen_parser
p = create_qwen_parser()
with p:
    board = p.process_script_file('data/TaskA/06.txt')
    assert any(panel.negative_prompt for panel in board.panels), 'FAIL: no LLM negatives populated'
"

# 3. SD3.5 triple negatives passed
python -c "
import inspect
from diffusers import StableDiffusion3Pipeline
sig = inspect.signature(StableDiffusion3Pipeline.__call__)
for k in ['negative_prompt_2', 'negative_prompt_3']:
    assert k in sig.parameters, f'FAIL: {k} not in pipeline signature'
"

# 4. Full batch benchmark
python storygen/scripts/run_parallel.py --max-gpus 8 --output outputs/optimization_v1 --cases 01 06 20
# After run: verify all 3 cases succeeded, avg CLIP ≥ 0.30, avg consistency ≥ 0.30
python -c "
import json
with open('outputs/optimization_v1/parallel_summary.json') as f:
    data = json.load(f)
assert data['success'] == data['total_cases'], f'FAIL: {data[\"failed\"]} cases failed'
"
```
