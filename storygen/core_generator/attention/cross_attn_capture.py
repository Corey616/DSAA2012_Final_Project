"""
Custom cross-attention processor that captures attention probability maps.
Used for building per-character spatial masks for bounded self-attention.
"""

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from typing import Optional, Dict, List


class CapturingCrossAttentionProcessor(AttnProcessor2_0):
    """
    Cross-attention processor that saves attention probability maps.

    The standard AttnProcessor2_0 returns only the final hidden states (3D).
    This processor saves the attention_probs (4D) before the final projection,
    enabling spatial mask extraction for bounded self-attention.
    """

    def __init__(self):
        super().__init__()
        self.attention_probs: Optional[torch.Tensor] = None
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.capture_enabled: bool = False

    def enable_capture(self):
        self.capture_enabled = True

    def disable_capture(self):
        self.capture_enabled = False
        self.attention_probs = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_self_attn = encoder_hidden_states is None
        if is_self_attn:
            kv_hidden_states = hidden_states
        else:
            kv_hidden_states = encoder_hidden_states
            if attn.norm_cross:
                kv_hidden_states = attn.norm_encoder_hidden_states(kv_hidden_states)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Save attention probabilities for mask building (cross-attention only)
        if self.capture_enabled and not is_self_attn and encoder_hidden_states is not None:
            # attention_probs: [batch*heads, seq, tokens]
            # Save a copy detached from computation graph
            self.attention_probs = attention_probs.detach().cpu()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def install_cross_attn_capture(pipe) -> List[CapturingCrossAttentionProcessor]:
    """
    Replace all cross-attention (attn2) processors with CapturingCrossAttentionProcessor.

    Args:
        pipe: Diffusers pipeline with unet or transformer

    Returns:
        List of installed capture processors
    """
    capture_processors = []

    # Determine which module to patch
    if hasattr(pipe, 'unet'):
        module = pipe.unet
    elif hasattr(pipe, 'transformer'):
        module = pipe.transformer
    else:
        raise ValueError("Pipeline has neither unet nor transformer")

    processor_map = {}
    for name, existing in module.attn_processors.items():
        if name.endswith('attn2.processor') or name.endswith('attn2'):
            capture = CapturingCrossAttentionProcessor()
            processor_map[name] = capture
            capture_processors.append(capture)
        else:
            processor_map[name] = existing

    module.set_attn_processor(processor_map)
    return capture_processors


def extract_character_masks(
    capture_processors: List[CapturingCrossAttentionProcessor],
    num_frames: int,
    text_encoder_max_length: int = 77,
    threshold: float = 0.3,
) -> Optional[torch.Tensor]:
    """
    Build bounded self-attention masks from captured cross-attention maps.

    For multi-character stories, characters mentioned in the prompt have distinct
    spatial positions in cross-attention maps. This function:
    1. Averages attention maps across all cross-attention layers
    2. Identifies high-attention regions for each text token position
    3. Creates masks where tokens attending to the SAME text region can attend to each other

    Returns:
        masks tensor or None if insufficient data
    """
    if not capture_processors:
        return None

    # Collect attention maps from all processors
    maps = []
    for cp in capture_processors:
        if cp.attention_probs is not None:
            # [batch*heads, seq, text_tokens] -> average over heads
            # batch = num_frames * 2 (cond + uncond)
            # Take only cond half
            attn = cp.attention_probs
            n_heads = attn.shape[0] // (num_frames * 2)
            attn_cond = attn[:num_frames * n_heads]  # First half
            attn_cond = attn_cond.view(num_frames, n_heads, -1, attn_cond.shape[-1])
            attn_cond = attn_cond.mean(dim=1)  # [num_frames, seq, text_tokens]
            maps.append(attn_cond)

    if not maps:
        return None

    # Average across all layers
    avg_maps = torch.stack(maps).mean(dim=0)  # [num_frames, seq, text_tokens]

    # For multi-frame, multi-character stories, build cross-frame attention masks
    # Mask[i, s, j, t] = similarity between what token s in frame i and token t in frame j
    # attend to in the text

    # Simplified: compute pairwise spatial overlap between frames
    # High overlap = same region = same character = allow attention
    num_frames = avg_maps.shape[0]
    seq_len = avg_maps.shape[1]

    # Normalize maps per frame
    maps_norm = torch.nn.functional.normalize(avg_maps, dim=-1)  # [F, S, T]

    # Compute cross-frame token similarity: [F, S, F, S]
    masks = torch.bmm(maps_norm.view(-1, seq_len, maps_norm.shape[-1]),
                      maps_norm.view(-1, seq_len, maps_norm.shape[-1]).transpose(-1, -2))
    masks = masks.view(num_frames, seq_len, num_frames, seq_len)

    # Threshold to binary: > 0.3 means same region (allow attention)
    masks = (masks > threshold).bool()

    # Always allow self-attention (same token position, same frame)
    for i in range(num_frames):
        masks[i, :, i, :] = True

    return masks
