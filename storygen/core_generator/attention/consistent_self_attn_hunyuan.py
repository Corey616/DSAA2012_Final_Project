"""
HunyuanDiT Consistent Self-Attention Processor
Adapted for HunyuanDiT's 40-block transformer with RoPE (image_rotary_emb).

Key differences from SDXL SCA:
- Processes installed on pipe.transformer.blocks (not pipe.unet)
- Handles image_rotary_emb (RoPE) with proper tiling for expanded K/V
- 40 blocks mapped to early(0-13)/mid(14-26)/late(27-39) tiers
- Uses HunyuanAttnProcessor2_0 base (head reshape, QK norm, SDP, RoPE)
- No attn.head_to_batch_dim — Hunyuan uses manual [B, H, S, D_head] reshape

SCA Invariants:
- Q unchanged: standard attention computation
- K/V expanded via windowed neighbor concatenation (truncated windows)
- RoPE applied AFTER K/V expansion (tiled for expanded sequence)
- Self-attention only (attn1 processors only)
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from diffusers.models.attention_processor import Attention, HunyuanAttnProcessor2_0
from diffusers.models.embeddings import apply_rotary_emb


class HunyuanSCAProcessor(HunyuanAttnProcessor2_0):
    """
    Consistent self-attention for HunyuanDiT transformer.

    HunyuanDiT uses self-attention (attn1) with RoPE position embeddings.
    SCA works by expanding K/V across frames, then applying RoPE to the
    concatenated K/V with proper tiling to match the expanded sequence length.
    """

    def __init__(
        self,
        consistency_strength: float = 0.6,
        window_size: int = 2,
        block_index: int = 0,
        total_blocks: int = 40,
    ):
        super().__init__()
        self.consistency_strength = consistency_strength
        self.window_size = window_size
        self.block_index = block_index
        self.total_blocks = total_blocks

        # Map block index to tier for layer-wise strength
        ratio = block_index / total_blocks
        if ratio < 0.35:
            self.layer_type = "early"
        elif ratio < 0.65:
            self.layer_type = "mid"
        else:
            self.layer_type = "late"

        self.layer_strength = {
            "early": min(consistency_strength * 0.75, 0.14),  # Layout/coarse structure
            "mid": min(consistency_strength * 1.5, 0.20),     # Identity/appearance
            "late": 0.0,                                         # Details — disabled to prevent blur
        }[self.layer_type]

        self.apply_after_ratio = 0.10
        self.current_step = 0
        self.total_steps = 1
        self.frame_pair_weights: Optional[List[List[float]]] = None

    def clear_memory(self):
        pass

    def set_step_state(self, current_step: int, total_steps: int):
        self.current_step = max(0, int(current_step))
        self.total_steps = max(1, int(total_steps))

    def set_story_context(self, frame_pair_weights: Optional[List[List[float]]]):
        self.frame_pair_weights = frame_pair_weights

    def set_bounded_masks(self, masks: Optional[torch.Tensor]):
        pass  # TODO: implement bounded attention for Hunyuan

    def _sca_active(self) -> bool:
        if self.total_steps <= 1:
            return True
        return (self.current_step / max(self.total_steps - 1, 1)) >= self.apply_after_ratio

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            kv_hidden_states = hidden_states
            is_self_attn = True
        else:
            kv_hidden_states = encoder_hidden_states
            if attn.norm_cross:
                kv_hidden_states = attn.norm_encoder_hidden_states(kv_hidden_states)
            is_self_attn = False

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        # ===== SCA: Cross-frame K/V concatenation =====
        if is_self_attn and batch_size > 1 and self.layer_strength > 0 and self._sca_active():
            num_stories = batch_size // 2
            inner_dim = key.shape[-1]

            k_cond, k_uncond = key.chunk(2, dim=0)
            v_cond, v_uncond = value.chunk(2, dim=0)

            # Windowed K/V concatenation with fixed-size windows (self-copy padding at edges)
            ws = min(self.window_size, num_stories - 1)

            def build_windows(sk, sv):
                k_windows, v_windows = [], []
                for i in range(num_stories):
                    k_parts, v_parts = [], []
                    for offset in range(-ws, ws + 1):
                        j = i + offset
                        if 0 <= j < num_stories:
                            part_k = sk[j:j + 1]
                            part_v = sv[j:j + 1]
                        else:
                            # Self-copy padding at edges
                            part_k = sk[i:i + 1]
                            part_v = sv[i:i + 1]
                        # Apply pair weights to non-self frames
                        if j != i and 0 <= j < num_stories:
                            # Apply pair_weight + layer_strength (same as SDXL SCA)
                            pw = 1.0
                            if self.frame_pair_weights is not None and len(self.frame_pair_weights) > i and len(self.frame_pair_weights[i]) > j:
                                pw = self.frame_pair_weights[i][j]
                            w = max(0.02, min(1.0, self.layer_strength * pw))
                            part_k = part_k * w
                            part_v = part_v * w
                        k_parts.append(part_k)
                        v_parts.append(part_v)
                    k_windows.append(torch.cat(k_parts, dim=0).unsqueeze(0))
                    v_windows.append(torch.cat(v_parts, dim=0).unsqueeze(0))
                return (
                    torch.cat(k_windows, dim=0).reshape(num_stories, -1, inner_dim),
                    torch.cat(v_windows, dim=0).reshape(num_stories, -1, inner_dim),
                )

            k_cond, v_cond = build_windows(k_cond, v_cond)
            k_uncond, v_uncond = build_windows(k_uncond, v_uncond)

            key = torch.cat([k_cond, k_uncond], dim=0)
            value = torch.cat([v_cond, v_uncond], dim=0)

        # Head reshape (Hunyuan style: [B, S, D] → [B, H, S, D_head])
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # QK normalization (HunyuanDiT specific)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE after SCA K/V expansion (HunyuanDiT specific)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if is_self_attn:
                # For SCA-expanded K/V, tile RoPE to match expanded sequence length
                cos, sin = image_rotary_emb  # each [S, D_head]
                if key.shape[2] != cos.shape[0]:
                    # Tile: the expanded sequence is a concatenation of neighbor frames
                    # each with the same spatial token ordering
                    tiled_cos = cos.repeat(key.shape[2] // cos.shape[0] + 1, 1)[:key.shape[2]]
                    tiled_sin = sin.repeat(key.shape[2] // sin.shape[0] + 1, 1)[:key.shape[2]]
                    key = apply_rotary_emb(key, (tiled_cos, tiled_sin))
                else:
                    key = apply_rotary_emb(key, image_rotary_emb)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Head reshape back: [B, H, S, D_head] → [B, S, D]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
