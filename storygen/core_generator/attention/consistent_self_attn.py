"""
Consistent Self-Attention Mechanism
StoryDiffusion-style implementation for training-free identity consistency

This module implements attention-based mechanisms to maintain character
consistency across multiple generated images without requiring model fine-tuning.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from diffusers.models.attention_processor import Attention, AttnProcessor


class ConsistentSelfAttentionProcessor(AttnProcessor):
    """
    StoryDiffusion-style consistent self-attention processor with:
    - Window attention: each frame attends to ±K neighbors (not all frames)
    - Layer-wise strength: shallow=scene consistency, deep=detail diversity
    """

    def __init__(
        self,
        consistency_strength: float = 0.6,
        memory_bank_size: int = 4,
        device: str = "cuda",
        layer_type: str = "mid_block",  # "down_blocks", "mid_block", "up_blocks"
        window_size: int = 2,  # ±K frames to attend to
        apply_after_ratio: float = 0.35,
    ):
        self.consistency_strength = consistency_strength
        self.memory_bank_size = memory_bank_size
        self.device = device
        self.layer_type = layer_type
        self.window_size = window_size
        self.apply_after_ratio = {
            "down_blocks": max(apply_after_ratio - 0.15, 0.15),
            "mid_block": max(apply_after_ratio - 0.22, 0.08),
            "up_blocks": min(apply_after_ratio + 0.05, 0.45),
        }.get(layer_type, apply_after_ratio)
        self.current_step = 0
        self.total_steps = 1

        # Layer-wise strength multipliers
        self.layer_strength = {
            "down_blocks": min(consistency_strength * 0.75, 0.14),  # Recover coarse identity anchors
            "mid_block": min(consistency_strength * 1.5, 0.24),     # Strongest identity/style consistency
            "up_blocks": min(consistency_strength * 0.35, 0.07),    # Keep detail sharing light
        }.get(layer_type, consistency_strength)

        self.memory_bank = []
        self.feature_projector = None
        self._projector_target_dim = None
        self.frame_pair_weights: Optional[List[List[float]]] = None

        # Bounded cross-frame attention (Storybooth-style)
        # Masks: [num_stories, seq_len, num_stories, seq_len] boolean tensor
        # masks[i, token_s, j, token_t] = True if token_s in frame i
        # can attend token_t in frame j (same character region)
        self.bounded_attention_masks: Optional[torch.Tensor] = None
        self.bounded_attention_active: bool = False

        # Cross-frame token merging (Storybooth-style)
        self.token_merging_enabled: bool = True  # Disabled by default (was True)
        self.token_merge_threshold: float = 0.75
        self.token_merge_fine_threshold: float = 0.55
        self.token_merge_weight: float = 0.5

    def clear_memory(self):
        self.memory_bank = []

    def set_step_state(self, current_step: int, total_steps: int):
        """Update denoising progress for step-aware SCA gating."""
        self.current_step = max(0, int(current_step))
        self.total_steps = max(1, int(total_steps))

    def set_story_context(self, frame_pair_weights: Optional[List[List[float]]]):
        """Attach per-frame cross-panel affinity weights for adaptive SCA."""
        self.frame_pair_weights = frame_pair_weights

    def set_bounded_masks(self, masks: Optional[torch.Tensor]):
        """Set per-character bounded attention masks (Storybooth-style).

        Args:
            masks: Boolean tensor [num_stories, seq_len, num_stories, seq_len]
                   masks[i, token_s, j, token_t] = True if token_s in frame i
                   can attend to token_t in frame j (same character region).
                   Pass None to disable bounded attention.
        """
        self.bounded_attention_masks = masks
        self.bounded_attention_active = masks is not None

    def _apply_token_merging(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Apply cross-frame token merging to align fine-grain features.

        Storybooth-style: merge similar tokens across frames to ensure
        consistent appearance details (clothing, hair, etc.).

        Args:
            hidden_states: Output from attention [batch_size * frames, seq_len, dim]
            batch_size: Original batch size (includes cond + uncond)

        Returns:
            hidden_states with tokens merged across frames
        """
        if not self.token_merging_enabled or batch_size <= 2:
            return hidden_states

        num_stories = batch_size // 2
        if num_stories <= 1:
            return hidden_states

        # Process cond and uncond separately
        cond, uncond = hidden_states.chunk(2, dim=0)

        def merge_frames(tensor: torch.Tensor) -> torch.Tensor:
            """Merge tokens across frames for one CFG half."""
            n_frames = tensor.shape[0]
            # Ensure we don't modify in-place
            result = tensor.clone()

            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    # Compute cosine similarity between tokens in frame i and frame j
                    sim = torch.mm(
                        F.normalize(result[i], dim=-1),
                        F.normalize(result[j], dim=-1).T,
                    )

                    # Find pairs above threshold
                    max_sim_i, max_idx_j = sim.max(dim=-1)

                    # Create merge mask for high-similarity pairs
                    merge_mask = max_sim_i > self.token_merge_threshold
                    if not merge_mask.any():
                        continue

                    # For each highly-similar token, blend it with its best match
                    src_indices = torch.where(merge_mask)[0]
                    tgt_indices = max_idx_j[src_indices]

                    for s, t in zip(src_indices.tolist(), tgt_indices.tolist()):
                        if s == t:
                            continue
                        # Weighted blending: merge_weight * target + (1-merge_weight) * source
                        blended = (
                            self.token_merge_weight * result[j, t]
                            + (1.0 - self.token_merge_weight) * result[i, s]
                        )
                        result[i, s] = blended
                        result[j, t] = blended

            return result

        cond = merge_frames(cond)
        uncond = merge_frames(uncond)

        return torch.cat([cond, uncond], dim=0)

    def _sca_active(self) -> bool:
        """Delay cross-frame mixing until after the coarse layout has formed."""
        if self.total_steps <= 1:
            return True
        return (self.current_step / max(self.total_steps - 1, 1)) >= self.apply_after_ratio

    def _pair_weight(self, source_index: int, target_index: int, total_frames: int) -> float:
        """Get the adaptive cross-frame weight for one frame pair."""
        if source_index == target_index:
            return 1.0
        if (
            self.frame_pair_weights
            and len(self.frame_pair_weights) == total_frames
            and len(self.frame_pair_weights[source_index]) == total_frames
        ):
            return float(max(0.0, min(1.0, self.frame_pair_weights[source_index][target_index])))
        return 1.0

    def update_memory(self, new_features: torch.Tensor):
        if len(self.memory_bank) >= self.memory_bank_size:
            self.memory_bank.pop(0)
        new_features = new_features.detach().to(dtype=torch.float16)
        self.memory_bank.append(new_features)

    def get_context_features(self) -> Optional[torch.Tensor]:
        if not self.memory_bank:
            return None
        stacked = torch.stack(self.memory_bank, dim=0)
        context = stacked.mean(dim=0)
        return context

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
            num_stories = batch_size // 2
            seq_kv, dim_kv = key.shape[1], key.shape[2]

            k_cond, k_uncond = key.chunk(2, dim=0)
            v_cond, v_uncond = value.chunk(2, dim=0)

            # Window attention: each frame attends to ±window_size neighbors.
            # Truncated windows — no self-copy padding for edge frames.
            ws = min(self.window_size, num_stories - 1)
            def build_weighted_windows(source_k: torch.Tensor, source_v: torch.Tensor):
                k_windows, v_windows = [], []
                for i in range(num_stories):
                    start = max(0, i - ws)
                    end = min(num_stories, i + ws + 1)
                    # Truncated window — no self-copy padding
                    k_w = source_k[start:end].unsqueeze(0)
                    v_w = source_v[start:end].unsqueeze(0)

                    # Apply pair weights to non-self frames
                    if self.frame_pair_weights is not None:
                        for offset_idx, j in enumerate(range(start, end)):
                            if j != i:
                                pw = self._pair_weight(i, j, num_stories)
                                w = pw  # pair_weight only (layer_strength removed — restores baseline behavior)
                                if w != 1.0:
                                    k_w[:, offset_idx:offset_idx+1] = k_w[:, offset_idx:offset_idx+1] * w
                                    v_w[:, offset_idx:offset_idx+1] = v_w[:, offset_idx:offset_idx+1] * w

                    k_windows.append(k_w)
                    v_windows.append(v_w)
                return (
                    torch.cat(k_windows, dim=0).reshape(num_stories, -1, dim_kv),
                    torch.cat(v_windows, dim=0).reshape(num_stories, -1, dim_kv),
                )

            k_cond, v_cond = build_weighted_windows(k_cond, v_cond)
            k_uncond, v_uncond = build_weighted_windows(k_uncond, v_uncond)

            key = torch.cat([k_cond, k_uncond], dim=0)
            value = torch.cat([v_cond, v_uncond], dim=0)
            attention_mask = None

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Apply cross-frame token merging for fine-grain consistency
        if encoder_hidden_states is None and batch_size > 1 and self.token_merging_enabled:
            hidden_states = self._apply_token_merging(hidden_states, batch_size)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RegionDisentangledProcessor(ConsistentSelfAttentionProcessor):
    """
    ReDiStory-style region disentangled processor

    Core improvement over base implementation:
    - Explicitly decomposes features into identity-related and scene-specific components
    - Applies different strategies at different layers
    - More fine-grained control over consistency vs diversity balance
    """

    def __init__(
        self,
        identity_weight: float = 0.7,
        scene_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.identity_weight = identity_weight
        self.scene_weight = scene_weight

        # Identity projection layers (learn to map features to identity subspace)
        hidden_dim = kwargs.get('dim', 320)
        self.identity_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.scene_projection = torch.nn.Linear(hidden_dim, hidden_dim)

    def disentangle_features(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feature disentanglement: Separate identity and scene components

        Args:
            features: Input feature tensor

        Returns:
            Tuple of (identity_features, scene_features)
        """
        identity = self.identity_projection(features)
        scene = self.scene_projection(features)

        # Orthogonality constraint can be added here for better separation
        # For simplicity, using direct decomposition

        return identity, scene

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        """
        Forward pass with feature disentanglement
        """
        # First execute standard consistent attention
        output = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Apply disentanglement recombination
        identity_feat, scene_feat = self.disentangle_features(output)

        # Weighted combination: stronger identity for consistency
        recombined = identity_feat * self.identity_weight + scene_feat * self.scene_weight

        return recombined


class ICSA_RACA_Processor:
    """
    TaleDiffusion-style Identity-Consistent Self-Attention and
    Region-Aware Cross-Attention processor

    Designed for multi-character scenes with spatial layout control:
    - ICSA: Maintains identity consistency for each character
    - RACA: Controls spatial relationships between characters
    """

    def __init__(
        self,
        num_characters: int = 2,
        spatial_weights: dict = None,
        **kwargs
    ):
        """
        Initialize multi-character attention processor

        Args:
            num_characters: Maximum number of characters to track
            spatial_weights: Dict mapping character pairs to attention weights
        """
        self.num_characters = num_characters
        self.spatial_weights = spatial_weights or {}
        super().__init__(**kwargs)

        # Character-specific feature banks
        self.character_features = {}

    def register_character(self, character_id: str, features: torch.Tensor):
        """Register a character's feature vector"""
        self.character_features[character_id] = features.detach().clone()

    def get_character_attention(
        self,
        query: torch.Tensor,
        character_id: str
    ) -> torch.Tensor:
        """
        Compute attention scores for a specific character

        Args:
            query: Query tensor
            character_id: Character identifier

        Returns:
            Attention scores for this character
        """
        if character_id not in self.character_features:
            return query

        char_features = self.character_features[character_id]

        # Compute similarity between query and character features
        # Simplified: using dot product attention
        similarity = torch.matmul(query, char_features.transpose(-2, -1))

        return similarity

    def apply_spatial_constraints(
        self,
        attention_scores: torch.Tensor,
        spatial_layout: dict
    ) -> torch.Tensor:
        """
        Apply spatial constraints to attention scores

        Args:
            attention_scores: Base attention scores
            spatial_layout: Dict describing relative positions

        Returns:
            Modified attention scores with spatial constraints
        """
        # Apply spatial weights based on layout
        for char_pair, weight in self.spatial_weights.items():
            # Parse character pair (e.g., "char1_char2")
            chars = char_pair.split("_")
            if len(chars) == 2 and all(c in self.character_features for c in chars):
                # Adjust attention based on spatial relationship
                attention_scores = attention_scores * weight

        return attention_scores
