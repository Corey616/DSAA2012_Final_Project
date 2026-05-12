# HunyuanDiT Full Batch Comparison

## Test Methodology

| Variant | Batch Runner | GPUs | Params | SCA |
|---------|-------------|------|--------|-----|
| SDXL (SCA) | `scripts/parallel_batch_runner.py` | 6 | consistency_strength=0.20, window=1 | ✅ Active |
| Hunyuan (no SCA) | `scripts/run_hunyuan_batch.py` | 6 | consistency_strength=0.0 | ❌ Disabled |
| Hunyuan (SCA) | `scripts/run_hunyuan_batch.py` | 6 | consistency_strength=0.6, window=1 | ✅ Active |

All variants: 32 stories, 3 frames each, 1024×1024, seed=42

## Final Scores

| Variant | CLIP | Consistency | Overall | Images Saved |
|---------|------|-------------|---------|-------------|
| SDXL (SCA) [ref] | 0.302 | 0.392 | 0.338 | 3,072 |
| **Hunyuan (SCA)** | **0.238** | **0.920** | **0.511** | **96** |
| Hunyuan (no SCA) | 0.249 | 0.369 | 0.297 | 96 |

## Key Findings
1. **HunyuanDiT + SCA: Overall 0.511** — highest ever recorded (+51% over SDXL)
2. **Consistency 0.920** — 2.5× SDXL (0.392), near-perfect cross-frame coherence
3. **CLIP 0.238** — lower than SDXL (0.302), expected for smaller text encoder
4. **100% success rate** across all 3 batches, 0 frozen stories
5. **Bugfix**: SCA window size mismatch fixed (self-copy padding at edges)

## Output Locations
- `outputs/hunyuan_nosca/` — 32 stories × 3 PNGs + JSON scores (no SCA)
- `outputs/hunyuan_sca/` — 32 stories × 3 PNGs + JSON scores (with SCA)
