# HunyuanDiT vs SDXL — A/B Comparison

## 3-Story Comparison (01, 03, 11)

| Model | Avg CLIP | Avg Consistency | Avg Overall |
|-------|----------|----------------|-------------|
| **SDXL** | 0.317 | 0.400 | 0.350 |
| **HunyuanDiT** | 0.287 | **0.662** | **0.437** |
| Δ | -0.030 (-9.5%) | +0.262 (+65.5%) | **+0.087 (+24.9%)** |

## Key Findings
1. **HunyuanDiT wins on overall** by +0.087 (24.9%)
2. **Consistency massively higher**: 0.662 vs 0.400 (+65.5%) even WITHOUT SCA
3. **CLIP lower**: 0.287 vs 0.317 (-9.5%) due to smaller text encoder
4. **HunyuanDiT's native self-attention** produces far more coherent cross-frame results
5. With SCA enabled, consistency could go even higher

## Recommendation
HunyuanDiT v1.2 is the best transformer backbone tested so far. Next steps:
1. Test with SCA enabled (consistency_strength > 0)
2. Tune generation params (steps, guidance)
3. Full 32-story batch comparison
