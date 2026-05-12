"""
PixArt-adapted Consistent Self-Attention Processor.
Same SCA principles as SDXL version but adapted for PixArt's transformer architecture.

Key differences from SDXL SCA:
- PixArt uses BasicTransformerBlock flat list (28 blocks), not UNet down/mid/up
- No CFG cond/uncond split needed (PixArt handles CFG differently)
- Layer-wise: divide 28 blocks into early(0-9)/mid(10-18)/late(19-27) groups
- Hidden states are patch embeddings [B, seq, dim], not spatial feature maps
- Uses the same AttnProcessor2_0 base class

SCA Invariants:
- Q unchanged: standard attention computation
- K/V expanded: [B, seq, dim] -> [B, B*seq, dim] via view+expand
- Window attention: each frame attends to +/-K neighbors
- 3D baddbmm only (same as SDXL)
- Self-attention only (attn1 processors only)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


class PixArtConsistentSelfAttentionProcessor(AttnProcessor2_0):
    def __init__(
        self,
        consistency_strength: float = 0.6,
        device: str = "cuda",
        layer_group: str = "mid",  # "early", "mid", "late"
        window_size: int = 2,
        apply_after_ratio: float = 0.25,
    ):
        self.consistency_strength = consistency_strength
        self.device = device
        self.layer_group = layer_group
        self.window_size = window_size
        self.apply_after_ratio = {
            "early": max(apply_after_ratio - 0.10, 0.15),
            "mid": max(apply_after_ratio - 0.15, 0.10),
            "late": min(apply_after_ratio + 0.05, 0.40),
        }.get(layer_group, apply_after_ratio)
        self.current_step = 0
        self.total_steps = 1

        self.layer_strength = {
            "early": min(consistency_strength * 0.75, 0.14),
            "mid": min(consistency_strength * 1.5, 0.24),
            "late": min(consistency_strength * 0.35, 0.07),
        }.get(layer_group, consistency_strength)

        self.frame_pair_weights: Optional[List[List[float]]] = None

    def set_step_state(self, current_step: int, total_steps: int):
        self.current_step = max(0, int(current_step))
        self.total_steps = max(1, int(total_steps))

    def set_story_context(self, frame_pair_weights: Optional[List[List[float]]]):
        self.frame_pair_weights = frame_pair_weights

    def _sca_active(self) -> bool:
        if self.total_steps <= 1:
            return True
        return (self.current_step / max(self.total_steps - 1, 1)) >= self.apply_after_ratio

    def _pair_weight(self, source_index: int, target_index: int, total_frames: int) -> float:
        if source_index == target_index:
            return 1.0
        if (
            self.frame_pair_weights
            and len(self.frame_pair_weights) == total_frames
            and len(self.frame_pair_weights[source_index]) == total_frames
        ):
            return float(max(0.0, min(1.0, self.frame_pair_weights[source_index][target_index])))
        return 1.0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states_for_kv = hidden_states
        else:
            encoder_hidden_states_for_kv = encoder_hidden_states
            if attn.norm_cross:
                encoder_hidden_states_for_kv = attn.norm_encoder_hidden_states(encoder_hidden_states_for_kv)

        key = attn.to_k(encoder_hidden_states_for_kv)
        value = attn.to_v(encoder_hidden_states_for_kv)

        # ===== SCA with Window Attention + Layer-wise Strength =====
        if encoder_hidden_states is None and batch_size > 1 and self.layer_strength > 0 and self._sca_active():
            num_frames = batch_size
            seq_kv, dim_kv = key.shape[1], key.shape[2]

            ws = min(self.window_size, num_frames - 1)

            def build_weighted_windows(source_k: torch.Tensor, source_v: torch.Tensor):
                k_windows, v_windows = [], []
                for i in range(num_frames):
                    k_parts, v_parts = [], []
                    for offset in range(-ws, ws + 1):
                        j = i + offset
                        if 0 <= j < num_frames:
                            part_k = source_k[j:j + 1]
                            part_v = source_v[j:j + 1]
                            pair_weight = self._pair_weight(i, j, num_frames)
                            weight = 1.0 if j == i else self.layer_strength * pair_weight
                        else:
                            part_k = source_k[i:i + 1]
                            part_v = source_v[i:i + 1]
                            weight = 1.0
                        if weight != 1.0:
                            part_k = part_k * weight
                            part_v = part_v * weight
                        k_parts.append(part_k)
                        v_parts.append(part_v)
                    k_windows.append(torch.cat(k_parts, dim=0).unsqueeze(0))
                    v_windows.append(torch.cat(v_parts, dim=0).unsqueeze(0))
                return (
                    torch.cat(k_windows, dim=0).reshape(num_frames, -1, dim_kv),
                    torch.cat(v_windows, dim=0).reshape(num_frames, -1, dim_kv),
                )

            key, value = build_weighted_windows(key, value)
            attention_mask = None

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
