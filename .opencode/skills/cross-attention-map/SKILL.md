# Skill: cross-attention-map

## Purpose
Extract cross-attention maps from HunyuanDiT transformer blocks to infer spatial character positions. Used as fallback when story scripts lack explicit spatial keywords ("left", "right", etc.).

## When to use
- When position masks need to be built but story scripts have no spatial keywords
- For multi-character stories (03, 06, 07) where character layout improves consistency
- After implementing the cross-attention extraction infrastructure

## Background

Current position masks require spatial keywords in the script text (`left`, `right`, `center`).
Only 5/32 stories have these keywords. For the other 27 stories, position masks return None.
Cross-attention maps from the DiT transformer show which spatial regions each character's
text tokens attend to, providing a layout signal without any script changes.

## Implementation Plan

### Phase 1: Extract Cross-Attention Maps
Add a hook to HunyuanDiT transformer blocks that captures attention_probabilities during
the forward pass for cross-attention layers (where encoder_hidden_states is not None).

### Phase 2: Infer Spatial Layouts
For each character mention in the prompt:
1. Find the character's token indices in the CLIP-encoded prompt
2. Extract cross-attention probabilities for those tokens from the cross-attention layers
3. Average across heads and layers
4. Find the spatial centroid (argmax of attention map)
5. Map centroid position to: "left" (x < 0.33), "center" (0.33 <= x < 0.66), "right" (x >= 0.66)

### Phase 3: Build Position Masks
Use the inferred positions to call the existing `_build_position_masks` in hunyuan_pipeline.py.
This reuses all the existing bounded attention infrastructure.

## Code Location
- Cross-attention hook: needs to be added to HunyuanDiT transformer blocks
- Position mask builder: already exists in `hunyuan_pipeline.py:_build_position_masks` (line ~135)
- Bounded attention: already exists in `HunyuanSCAProcessor.set_bounded_masks`

## Key Considerations
- **Batch dimension**: Cross-attention maps have batch dimension [B, H, S, S_k] — need to handle unconditional batch half
- **Layer selection**: Transformer-Decoder blocks 0-39 all have cross-attention; use mid-layers (14-26) for best spatial signal
- **Resolution**: Attention maps are at latent resolution (S = H_lat * W_lat = 64*64 for 1024px)
- **RoPE**: Cross-attention does NOT use RoPE (only self-attention does), so no tiling needed
- **Multiple characters**: When multiple characters attend to the same region, use the one with highest peak activation

## Verification
After implementation, verify on story 03 (multi-character: Dog + Cat) that:
1. Cross-attention maps show distinct spatial peaks for "Dog" vs "Cat" tokens
2. Inferred positions are consistent across frames 1-3
3. Position masks are successfully built and applied

## Output
Provide the orchestrator with:
1. Inferred positions per character per frame
2. Whether position masks were successfully applied
3. Any stories where inference failed and why
