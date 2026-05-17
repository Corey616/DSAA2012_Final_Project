# StoryGen Work Memory — Lessons Learned

## Architecture (stable)
- **Backbone**: SD3.5-Medium (StableDiffusion3Pipeline), already cached
- **LLM**: Qwen3-4B-Instruct-2507, LLM-only (no rule-based fallbacks)
- **Style**: Anime/Ghibli — flat colors mask texture/pattern inconsistencies
- **Multi-GPU**: run_parallel.py, 8×A800, 2 tasks/GPU, round-robin distribution
- **CWD**: Always `storygen/`, imports use `from src.xxx` (renamable root)
- **Models**: All cached at `storygen/models/`

## What Works ✅
1. **Anime style baseline**: 32/32 cases, 8.9min, 2 tasks/GPU
2. **Color harmonization**: `match_histogram()` prevents desaturation
3. **LLM prompt refinement**: Section 6B/C for negative prompts + style lock (in SYSTEM_PROMPT, not yet wired)
4. **Portability**: `environment.yml` + `requirements.txt` exist
5. **Single-case script**: `run_worker.py --gpu N --script data/TaskA/XX.txt`
6. **Batch script**: `run_parallel.py --max-gpus 8`

## What Failed ❌
1. **Img2Img (T1)**: Gray frames on subsequent panels — REVERTED
2. **Prompt identity anchoring (T2)**: Pure gray frames 2+ — REVERTED
3. **CFG 5.0 + triple negative encoder**: Face collapse, animal breed shift, style drift — REVERTED
4. **SCA**: Dimension mismatch (VAE 4-dim vs UNet 1280-dim), disabled
5. **IP-Adapter**: Not tested — risk of scene structure freezing similar to img2img

## Design Principles
- **NO img2img** — binds scene structure, prevents scene transitions
- **NO SCA** — dimension mismatch, requires batch generation
- **IP-Adapter with caution** — may freeze identity/scene
- **CFG 7.5 for anime** — 5.0 is too low for SD3.5 anime style
- **Keep negative prompts simple** — triple encoder (`neg_prompt_2/_3`) destabilizes
- **Anime style is key** — masks texture/pattern drift, better cross-frame consistency
- **LLM-only parsing** — random character generation removed (~500 lines)

## Git History
```
6cb35a3 [REVERT] Rollback CFG 5.0 + triple neg encoder → v2 stable
0468b8b [REVERT+STYLE] Revert T1/T2, switch to anime style
213c3be [CONFIG] 2 tasks/GPU parallel
508be79 [OPT] Tier 1 optimizations (reverted T1/T2)
e8b7dbf [REFACTOR] Unified storygen/ directory structure
```

## Next Steps (not implemented yet)
1. Fix Panel.negative_prompt extraction from LLM JSON (field exists, not populated)
2. Style consistency enhancement (stronger anchoring in _compose_prompt)
3. AB test framework for comparing prompt variants
4. Action/scene alignment improvement
5. README.md for TA setup instructions
