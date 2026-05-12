# Transformer Backbone Alternatives — Research Results

## Tested: PixArt-Sigma (2026-05-11)
- SDXL vs PixArt A/B: SDXL=0.350, PixArt=0.327 (SDXL wins by +0.023)
- PixArt needs CFG-aware SCA for batch mode parity
- Pipeline committed (432 lines)

## Recommended: HunyuanDiT v1.2 (2026-05-11)
- **Architecture**: Pure DiT, 40 blocks, clean SA/CA separation (attn1/attn2 per block)
- **Params**: 1.50B (smaller than SDXL's 2.6B)
- **Memory**: ~14 GB for 3-frame batch (same as SDXL)
- **Resolution**: Native 1024×1024
- **SCA compatibility**: High — same attn1/attn2 pattern as SDXL
- **Status**: Downloaded and cached (14 GB)
- **Next**: Create SCA processor with RoPE handling, then A/B compare

## Evaluated but Not Recommended
| Model | Reason |
|-------|--------|
| SD3.5 Medium | Joint attention (no SA/CA separation) — SCA integration very complex |
| Flux.1-dev | 8B params, joint attention + RoPE — too large, too complex |
