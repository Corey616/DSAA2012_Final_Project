"""
Narrative Generation Pipeline - Core Story Generation Engine
Integrates all SOTA techniques into a unified generation interface
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import numpy as np
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup HF mirror for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from storygen.script_director.llm_parser import ProductionBoard, Panel
from storygen.asset_anchor.character_portrait import CharacterPortraitGenerator
from storygen.core_generator.attention.consistent_self_attn import ConsistentSelfAttentionProcessor
from storygen.core_generator.memory_bank import MemoryBank
from storygen.utils.image_utils import remove_white_borders


class NarrativeGenerationPipeline:
    """
    Main Story Generation Pipeline

    This class orchestrates the entire story generation process, including:
    - LLM-directed production board parsing
    - Character portrait generation and feature extraction
    - Consistent image generation with memory
    - Multi-frame story creation

    Usage:
        pipeline = NarrativeGenerationPipeline(config)
        images = pipeline.generate_story(production_board)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generation pipeline

        Args:
            config: Configuration dictionary containing:
                - base_model: Path to SDXL model
                - consistency_mode: "storydiffusion" | "redistory" | "hybrid"
                - device: Computation device
                - generation_params: Generation settings
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if config.get("use_fp16", True) else torch.float32

        # Component initialization flags
        self._initialized = False
        self._base_pipe = None
        self._portrait_gen = None
        self._attn_processor = None
        self._memory_bank = None
        self._identity_adapter_loaded = False
        self._identity_adapter_source = None
        self._runtime_optimizations_applied = False

        # Cross-attention map extraction for bounded SCA masks
        self._cross_attention_maps: Dict[str, torch.Tensor] = {}
        self._cross_attention_hooks: List = []
        self._cross_attn_capture_processors: List = []

        print("=" * 60)
        print("Narrative Weaver Pro - Generation Engine")
        print("=" * 60)

    @property
    def base_pipe(self):
        """Lazy load base diffusion pipeline with cache-first strategy"""
        if self._base_pipe is None:
            from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
            from storygen.utils.mirror_config import verify_model_integrity, get_models_cache_dir, get_sdxl_model_path

            model_name = self.config.get("base_model") or get_sdxl_model_path()
            cache_dir = get_models_cache_dir()
            print(f"[Pipeline] Loading SDXL Base Model: {model_name}")
            print(f"[Pipeline] Using cache directory: {cache_dir}")

            # Check cache integrity first
            is_complete = verify_model_integrity(model_name, cache_dir)

            if is_complete:
                print("[Pipeline] ✓ Using local cache (skip network verification)")
                load_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "variant": "fp16" if self.dtype == torch.float16 else None,
                    "local_files_only": True,  # Skip network verification
                    "low_cpu_mem_usage": True,  # Memory optimization
                    "cache_dir": str(cache_dir),  # Use project ./models directory
                }
            else:
                print("[Pipeline] ⚠ Cache incomplete/missing, downloading from mirror...")
                load_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "variant": "fp16" if self.dtype == torch.float16 else None,
                    "local_files_only": False,
                    "low_cpu_mem_usage": True,
                    "cache_dir": str(cache_dir),  # Use project ./models directory
                }

            self._base_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                **load_kwargs
            ).to(self.device)

            # Use DPM++ scheduler for faster convergence
            self._base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._base_pipe.scheduler.config
            )

        return self._base_pipe

    @property
    def portrait_gen(self):
        """Lazy load character portrait generator"""
        if self._portrait_gen is None:
            print("[Pipeline] Initializing Character Portrait Generator...")
            self._portrait_gen = CharacterPortraitGenerator(
                base_model=self.base_pipe,
                device=self.device,
                dtype=self.dtype
            )
        return self._portrait_gen

    @property
    def attn_processor(self):
        """Lazy load attention processor with window attention"""
        if self._attn_processor is None:
            consistency_strength = self.config.get("consistency_strength", 0.0)
            if consistency_strength > 0:
                window_size = self.config.get("sca_window_size", 1)
                start_ratio = self.config.get("sca_start_ratio", 0.35)
                print(f"[Pipeline] Setting up SCA (strength={consistency_strength}, window={window_size}, start={start_ratio})...")
                processor_map = {}
                sca_count = 0
                existing_processors = self.base_pipe.unet.attn_processors
                for name, existing_processor in existing_processors.items():
                    if name.endswith("attn1.processor"):
                        if name.startswith("down_blocks"):
                            layer_type = "down_blocks"
                        elif name.startswith("up_blocks"):
                            layer_type = "up_blocks"
                        else:
                            layer_type = "mid_block"
                        processor_map[name] = ConsistentSelfAttentionProcessor(
                            consistency_strength=consistency_strength,
                            memory_bank_size=self.config.get("memory_bank_size", 4),
                            device=self.device,
                            layer_type=layer_type,
                            window_size=window_size,
                            apply_after_ratio=start_ratio,
                        )
                        sca_count += 1
                    else:
                        processor_map[name] = existing_processor
                self._attn_processor = processor_map
                self.base_pipe.unet.set_attn_processor(self._attn_processor)
                print(f"[Pipeline] SCA applied to {sca_count} self-attention layers")
            else:
                print("[Pipeline] Using default attention (consistency disabled)")
                self._attn_processor = None
        return self._attn_processor

    @property
    def memory_bank(self):
        """Lazy load memory bank"""
        if self._memory_bank is None:
            print("[Pipeline] Initializing Memory Bank...")
            self._memory_bank = MemoryBank(
                capacity=self.config.get("memory_bank_capacity", 5),
                decay_factor=self.config.get("memory_decay_factor", 0.9),
                device=self.device
            )
        return self._memory_bank

    def initialize(self):
        """Explicit initialization of all components"""
        if self._initialized:
            return

        # Trigger lazy loading of all components
        _ = self.base_pipe
        _ = self.portrait_gen
        self._ensure_identity_adapter()
        _ = self.attn_processor
        self._configure_runtime_optimizations()
        _ = self.memory_bank

        self._initialized = True
        print("[Pipeline] All components initialized successfully!\n")

    def _get_story_panel_state(
        self,
        story_state: Optional[Dict[str, Any]],
        panel_id: int
    ) -> Dict[str, Any]:
        """Fetch the structured state block for one panel."""
        if not story_state:
            return {}
        for panel_state in story_state.get("panel_states", []):
            if panel_state.get("panel_id") == panel_id:
                return panel_state
        return {}

    def _set_sca_step_state(self, current_step: int, total_steps: int):
        """Propagate denoising progress into all SCA processors."""
        if isinstance(self._attn_processor, dict):
            for proc in self._attn_processor.values():
                if isinstance(proc, ConsistentSelfAttentionProcessor):
                    proc.set_step_state(current_step, total_steps)
        elif isinstance(self._attn_processor, ConsistentSelfAttentionProcessor):
            self._attn_processor.set_step_state(current_step, total_steps)

    def _normalize_story_label(self, text: str) -> str:
        """Normalize scene/layout labels for matching across panels."""
        import re
        normalized = str(text or "").lower().strip(" ,.")
        normalized = re.sub(r"^(in|at|inside|outside|interior of|exterior of)\s+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _configure_runtime_optimizations(self):
        """Apply memory/runtime knobs after adapter and attention processors are installed."""
        if self._runtime_optimizations_applied:
            return

        enable_offload = self.config.get("enable_model_cpu_offload", True)
        consistency = self.config.get("consistency_strength", 0.0)
        if enable_offload and consistency == 0:
            self.base_pipe.enable_model_cpu_offload()
            print("[Pipeline] CPU offload enabled")
        elif consistency > 0:
            if self._identity_adapter_loaded:
                print("[Pipeline] SCA + face lock active: skipping attention slicing to preserve IP-Adapter processors")
            else:
                print("[Pipeline] SCA active: enabling attention slicing after adapter setup")
                self.base_pipe.enable_attention_slicing()
        else:
            print("[Pipeline] GPU mode (no offload)")

        self._runtime_optimizations_applied = True

    def _get_ip_adapter_snapshot_dir(self) -> Optional[Path]:
        """Resolve the local IP-Adapter cache snapshot if available."""
        from storygen.utils.mirror_config import get_models_cache_dir

        cache_dir = get_models_cache_dir()
        snapshot_root = cache_dir / "models--h94--IP-Adapter" / "snapshots"
        if not snapshot_root.exists():
            return None

        snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
        return snapshots[0] if snapshots else None

    def _ensure_identity_adapter(self):
        """Identity adapter disabled for generalization baseline."""
        pass

    def _is_human_character(self, char_name: str, char_info: Any) -> bool:
        """Gate face-lock to human characters only."""
        text_parts = [char_name]
        for attr in ("visual_description", "appearance_details", "clothing"):
            value = getattr(char_info, attr, "")
            if value:
                text_parts.append(str(value))
        normalized = " ".join(text_parts).lower()
        non_human_keywords = {
            "robot", "android", "cyborg", "machine", "mechanical",
            "dog", "cat", "bird", "rabbit", "horse", "lion", "tiger", "bear",
            "wolf", "fox", "deer", "cow", "pig", "elephant", "monkey", "panda",
            "fish", "shark", "whale", "dolphin", "penguin", "duck", "eagle",
            "owl", "parrot", "snake", "lizard", "turtle", "frog", "dragon",
        }
        return not any(keyword in normalized for keyword in non_human_keywords)

    def _resolve_character_entry(
        self,
        char_name: str,
        characters: Dict[str, Any],
        portraits: Dict[str, Tuple[Image.Image, torch.Tensor]],
    ) -> Tuple[Optional[str], Optional[Any], Optional[Tuple[Image.Image, torch.Tensor]]]:
        """Resolve character and portrait entries case-insensitively."""
        lowered = str(char_name).lower()

        character_key = next((name for name in characters.keys() if str(name).lower() == lowered), None)
        portrait_key = next((name for name in portraits.keys() if str(name).lower() == lowered), None)

        if character_key is None or portrait_key is None:
            return None, None, None
        return character_key, characters[character_key], portraits[portrait_key]

    def _get_story_face_lock_character(
        self,
        production_board: ProductionBoard,
        portraits: Dict[str, Tuple[Image.Image, torch.Tensor]],
    ) -> Optional[str]:
        return None

    def _get_panel_identity_reference(
        self,
        panel: Panel,
        panel_state: Dict[str, Any],
        production_board: ProductionBoard,
        portraits: Dict[str, Tuple[Image.Image, torch.Tensor]],
        eligible_story_character: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], float, Optional[str]]:
        return None, 0.0, None

    def _build_batch_ip_adapter_embeds(
        self,
        identity_images: List[Image.Image],
        identity_scales: List[float],
    ) -> Optional[List[torch.Tensor]]:
        """Encode portrait references once and apply per-frame strength scaling."""
        if not self._identity_adapter_loaded:
            return None

        if not identity_images or max(identity_scales, default=0.0) <= 0:
            return None

        if getattr(self.base_pipe, "image_encoder", None) is None or getattr(self.base_pipe, "feature_extractor", None) is None:
            return None

        image_encoder = self.base_pipe.image_encoder
        encoder_dtype = next(image_encoder.parameters()).dtype
        image_tensor = self.base_pipe.feature_extractor(identity_images, return_tensors="pt").pixel_values
        image_tensor = image_tensor.to(device=self.device, dtype=encoder_dtype)

        with torch.no_grad():
            positive = image_encoder(image_tensor).image_embeds

        positive = positive.to(device=self.device, dtype=self.dtype)
        scale_tensor = torch.tensor(identity_scales, device=self.device, dtype=self.dtype).unsqueeze(-1)
        positive = positive * scale_tensor
        negative = torch.zeros_like(positive)
        return [torch.stack([negative, positive], dim=0).unsqueeze(2)]

    def _slice_ip_adapter_embeds(
        self,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]],
        frame_index: int,
        total_frames: int,
    ) -> Optional[List[torch.Tensor]]:
        """Extract one frame's conditioning embeds for sequential fallback."""
        if not ip_adapter_image_embeds:
            return None

        frame_embeds: List[torch.Tensor] = []
        for embed in ip_adapter_image_embeds:
            if embed.ndim >= 2 and embed.shape[0] == 2 and embed.shape[1] == total_frames:
                frame_embeds.append(embed[:, frame_index:frame_index + 1].contiguous())
            elif embed.ndim >= 1 and embed.shape[0] == total_frames * 2:
                negative = embed[frame_index:frame_index + 1]
                positive = embed[frame_index + total_frames:frame_index + total_frames + 1]
                frame_embeds.append(torch.cat([negative, positive], dim=0).contiguous())
            elif embed.ndim >= 1 and embed.shape[0] == total_frames:
                frame_embeds.append(embed[frame_index:frame_index + 1].contiguous())
        return frame_embeds or None

    def _build_sca_pair_weights(self, story_state: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
        """Build adaptive frame affinity weights from StoryState."""
        panel_states = (story_state or {}).get("panel_states", [])
        scene_slots = (story_state or {}).get("scene_slots", [])
        if not panel_states:
            return None

        weights: List[List[float]] = []
        for i, source in enumerate(panel_states):
            source_chars = {name.lower() for name in source.get("characters_present", [])}
            source_count = int(source.get("expected_count") or len(source_chars))
            source_scene = self._normalize_story_label(scene_slots[i] if i < len(scene_slots) else "")
            source_layout = self._normalize_story_label(source.get("spatial_layout", ""))
            source_shot = self._normalize_story_label((source.get("camera_plan") or {}).get("shot_type", ""))
            source_actions = {str(item).lower() for item in source.get("action_beats", [])}
            source_props = {
                str(item).lower()
                for item in source.get("must_show", [])
                if str(item).lower() not in source_chars
                and str(item).lower() not in source_actions
                and str(item).lower() != source_scene
                and str(item).lower() != source_layout
            }

            row: List[float] = []
            for j, target in enumerate(panel_states):
                if i == j:
                    row.append(1.0)
                    continue

                target_chars = {name.lower() for name in target.get("characters_present", [])}
                target_count = int(target.get("expected_count") or len(target_chars))
                target_scene = self._normalize_story_label(scene_slots[j] if j < len(scene_slots) else "")
                target_layout = self._normalize_story_label(target.get("spatial_layout", ""))
                target_shot = self._normalize_story_label((target.get("camera_plan") or {}).get("shot_type", ""))
                target_actions = {str(item).lower() for item in target.get("action_beats", [])}
                target_props = {
                    str(item).lower()
                    for item in target.get("must_show", [])
                    if str(item).lower() not in target_chars
                    and str(item).lower() not in target_actions
                    and str(item).lower() != target_scene
                    and str(item).lower() != target_layout
                }

                shared_chars = source_chars.intersection(target_chars)
                char_overlap = len(shared_chars) / max(1, max(len(source_chars), len(target_chars)))
                same_scene = bool(source_scene and source_scene == target_scene)
                adjacent = abs(i - j) == 1
                shared_props = source_props.intersection(target_props)

                weight = 0.18
                if char_overlap > 0:
                    weight += 0.50 * char_overlap
                if source_count == target_count and source_count > 1:
                    weight += 0.12
                if same_scene:
                    weight += 0.18
                elif adjacent and char_overlap > 0:
                    weight += 0.12
                if shared_props:
                    weight += 0.10
                if source_layout and target_layout and len(source_chars) > 1 and len(target_chars) > 1:
                    weight += 0.10
                if source_count != target_count:
                    weight -= float(self.config.get("sca_count_mismatch_penalty", 0.10))
                if self.config.get("enable_shot_type_sca_gating", False):
                    wide_like = {"wide", "establishing"}
                    if source_shot in wide_like or target_shot in wide_like:
                        weight -= float(self.config.get("sca_wide_shot_penalty", 0.0))
                    if source_count > 1 or target_count > 1:
                        weight -= float(self.config.get("sca_multi_character_penalty", 0.0))
                if source_scene and target_scene and source_scene != target_scene:
                    weight -= float(self.config.get("sca_scene_change_penalty", 0.0))
                if not shared_chars:
                    weight = min(weight, 0.20)
                    if source_count != target_count:
                        weight = min(weight, 0.10)

                min_weight = float(self.config.get("sca_min_pair_weight", 0.15))
                row.append(max(min_weight, min(0.92, weight)))
            weights.append(row)
        return weights

    def _set_sca_story_context(self, story_state: Optional[Dict[str, Any]]):
        """Propagate story-aware frame affinity weights into all SCA processors."""
        pair_weights = self._build_sca_pair_weights(story_state)
        if isinstance(self._attn_processor, dict):
            for proc in self._attn_processor.values():
                if isinstance(proc, ConsistentSelfAttentionProcessor):
                    proc.set_story_context(pair_weights)
        elif isinstance(self._attn_processor, ConsistentSelfAttentionProcessor):
            self._attn_processor.set_story_context(pair_weights)

    def _register_cross_attention_hooks(self):
        """Replace cross-attention (attn2) processors with CapturingCrossAttentionProcessor
        for attention probability map extraction used in bounded SCA masks."""
        from .attention.cross_attn_capture import install_cross_attn_capture

        # Save original attn2 processors for restoration during cleanup
        self._original_attn2_processors = {}
        if hasattr(self, '_base_pipe') and self._base_pipe is not None:
            for name, proc in self.base_pipe.unet.attn_processors.items():
                if name.endswith('attn2.processor') or name.endswith('attn2'):
                    self._original_attn2_processors[name] = proc

        # Install capturing processors
        self._cross_attn_capture_processors = install_cross_attn_capture(self.base_pipe)

        # Enable capture for all installed processors
        for cp in self._cross_attn_capture_processors:
            cp.enable_capture()

    def _build_character_masks_from_attention(
        self,
        prompts: List[str],
        height: int,
        width: int,
        production_board=None,  # Optional: provides authoritative character names
    ) -> Optional[torch.Tensor]:
        """Build per-character spatial masks from captured cross-attention maps.

        Uses the CapturingCrossAttentionProcessor instances installed by
        _register_cross_attention_hooks to extract attention probabilities.

        Returns:
            masks tensor [num_frames, seq_len, num_frames, seq_len] or None
        """
        if not self._cross_attn_capture_processors:
            return None

        # Extract character names from production_board (authoritative source)
        char_names = set()
        if production_board and hasattr(production_board, 'characters'):
            char_names = set(production_board.characters.keys())

        # Fallback: extract from prompts using the original heuristic
        if not char_names:
            for prompt in prompts:
                first_part = prompt.split(',')[0].strip().lower()
                if first_part:
                    char_names.add(first_part)

        if len(char_names) <= 1:
            return None  # Single-character stories don't need bounded attention

        from .attention.cross_attn_capture import extract_character_masks
        return extract_character_masks(
            self._cross_attn_capture_processors,
            num_frames=len(prompts),
            threshold=0.3,
        )

    def _cleanup_cross_attention_hooks(self):
        """Restore original cross-attention processors and clear captured data."""
        # Restore original attn2 processors
        if hasattr(self, '_original_attn2_processors') and self._original_attn2_processors:
            processor_map = dict(self.base_pipe.unet.attn_processors)
            for name, original_proc in self._original_attn2_processors.items():
                processor_map[name] = original_proc
            self.base_pipe.unet.set_attn_processor(processor_map)
        # Clear state
        self._cross_attn_capture_processors = []
        self._original_attn2_processors = {}
        self._cross_attention_maps = {}

    def _compile_prompt_from_slots(
        self,
        slots: List[Dict[str, Any]],
        max_len: int = 300
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Compile prioritized prompt slots into a bounded-length SDXL prompt."""
        slot_char_budgets = {
            "identity_anchor": 96,
            "count_anchor": 52,
            "action_anchor": 72,
            "scene_anchor": 72,
            "scene_continuity_anchor": 88,
            "layout_anchor": 60,
            "relation_anchor": 52,
            "expression_anchor": 36,
            "continuity_anchor": 64,
            "prop_anchor": 48,
            "camera_anchor": 36,
            "time_anchor": 24,
        }
        normalized_slots = []
        for slot in slots:
            text = self._clean_prompt_fragment(str(slot.get("text", "")))
            if not text:
                continue
            slot_name = slot.get("name", "slot")
            budget = slot_char_budgets.get(slot_name)
            if budget:
                text = self._truncate_prompt_fragment(text, budget)
            if not text:
                continue
            normalized_slots.append({
                "name": slot_name,
                "text": text,
                "priority": slot.get("priority", 0),
                "required": bool(slot.get("required", False)),
            })

        style_slot = None
        if normalized_slots and normalized_slots[-1]["name"] == "style_anchor":
            style_slot = normalized_slots.pop()

        required_slots = sorted(
            [slot for slot in normalized_slots if slot["required"]],
            key=lambda item: -item["priority"],
        )
        optional_slots = sorted(
            [slot for slot in normalized_slots if not slot["required"]],
            key=lambda item: -item["priority"],
        )

        selected_slots = []
        remaining = max_len - (len(style_slot["text"]) + 2 if style_slot else 0)
        remaining = max(remaining, 0)

        for slot in required_slots + optional_slots:
            extra = len(slot["text"]) + (2 if selected_slots else 0)
            if extra <= remaining:
                selected_slots.append(slot)
                remaining -= extra
                continue

            if slot["required"] and remaining > 12:
                truncated = slot["text"][:remaining].rsplit(" ", 1)[0].strip(" ,")
                if truncated:
                    selected_slots.append({**slot, "text": truncated, "truncated": True})
                    remaining = 0

        prompt_parts = [slot["text"] for slot in selected_slots]
        if style_slot:
            prompt_parts.append(style_slot["text"])

        return ", ".join(prompt_parts), selected_slots + ([style_slot] if style_slot else [])

    def _get_prompt_char_limit(self, binding_priority_panel: bool) -> int:
        """Only expand prompt budget for the narrow multi-human binding cases that need it."""
        return 344 if binding_priority_panel else 300

    def _needs_binding_priority_panel(
        self,
        panel: Panel,
        panel_state: Dict[str, Any],
        panel_entities: List[Dict[str, Any]],
        expected_count: int,
    ) -> bool:
        """Gate the larger prompt budget to multi-human pair-binding risk panels only."""
        if expected_count <= 1 or len(panel_entities) < 2:
            return False
        entity_types = {
            self._clean_prompt_fragment(entity.get("entity_type", "")).lower()
            for entity in panel_entities
            if self._clean_prompt_fragment(entity.get("entity_type", ""))
        }
        if entity_types != {"human"}:
            return False
        if "left/right" not in str(panel_state.get("spatial_layout", "")).lower():
            return False

        risk_text = " ".join(
            filter(
                None,
                [
                    panel.raw_prompt or "",
                    getattr(panel, "setting", "") or "",
                    " ".join(str(item) for item in panel_state.get("action_beats", [])),
                ],
            )
        ).lower()
        risk_tokens = ("crowd", "crowded", "market", "meet", "each other")
        return any(token in risk_text for token in risk_tokens)

    def _text_has_term(self, text: str, term: str) -> bool:
        """Match a single- or multi-word term using word boundaries."""
        import re

        cleaned_text = str(text or "").lower()
        cleaned_term = self._clean_prompt_fragment(term).lower()
        if not cleaned_text or not cleaned_term:
            return False
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in cleaned_term.split()) + r"\b"
        return bool(re.search(pattern, cleaned_text, flags=re.IGNORECASE))

    def _text_has_any_term(self, text: str, terms: List[str]) -> bool:
        """Return True when any whitelisted term appears in the text."""
        return any(self._text_has_term(text, term) for term in terms)

    def _build_action_support_relation(
        self,
        panel: Panel,
        panel_state: Dict[str, Any],
        action_text: str,
        scene_text: str,
        relation_text: str,
        panel_props: List[str],
        continuity_targets: List[str],
        primary_entity_type: str,
    ) -> Tuple[str, Optional[str]]:
        """Relation passthrough — no story-specific injection rules."""
        if int(panel_state.get("expected_count") or 0) != 1:
            return None, None
        return relation_text, None

    def _extend_travel_continuity_targets(
        self,
        story_state: Optional[Dict[str, Any]],
        panel_state: Dict[str, Any],
        panel: Panel,
        action_text: str,
        continuity_targets: List[str],
        panel_props: List[str],
        primary_entity_type: str,
    ) -> List[str]:
        """Travel continuity passthrough — no story-specific rules."""
        return continuity_targets

    def _clean_prompt_fragment(self, text: str) -> str:
        """Normalize prompt fragments and drop dangling trailing connectors."""
        import re

        cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" ,.")
        if not cleaned:
            return ""

        dangling_tokens = {
            "a", "an", "the", "and", "or", "with", "of", "to", "by", "near",
            "at", "in", "on", "over", "under", "from", "into", "onto", "for",
            "through", "around", "inside", "outside", "behind", "beside",
        }
        parts = cleaned.split()
        while parts and parts[-1].lower() in dangling_tokens:
            parts.pop()
        cleaned = " ".join(parts).strip(" ,.")
        return cleaned

    def _split_state_terms(self, text: str) -> List[str]:
        if not text:
            return []
        import re
        parts = re.split(r",|;|/|\band\b|\bwith\b", text)
        items = []
        seen = set()
        for part in parts:
            cleaned = re.sub(r"\s+", " ", part).strip(" .")
            if len(cleaned) < 3:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            items.append(cleaned)
        return items

    def _truncate_prompt_fragment(self, text: str, max_len: int) -> str:
        """Trim one fragment to a char budget without leaving broken endings."""
        text = self._clean_prompt_fragment(text)
        if len(text) <= max_len:
            return text
        shortened = text[:max_len].rsplit(" ", 1)[0]
        shortened = self._clean_prompt_fragment(shortened)
        if shortened:
            return shortened
        return self._clean_prompt_fragment(text[:max_len])

    def _join_segments_with_budget(self, parts: List[str], max_len: int) -> str:
        """Keep full descriptive segments when possible instead of mid-phrase truncation."""
        selected: List[str] = []
        for raw_part in parts:
            part = self._clean_prompt_fragment(raw_part)
            if not part:
                continue
            candidate = ", ".join(selected + [part]) if selected else part
            if len(candidate) <= max_len:
                selected.append(part)
                continue
            if not selected:
                trimmed = self._truncate_prompt_fragment(part, max_len)
                if trimmed:
                    selected.append(trimmed)
            break
        return ", ".join(selected)

    def _clean_identity_term(self, text: str, char_name: str) -> str:
        """Remove duplicated name/pronoun scaffolding from identity fragments."""
        import re

        cleaned = self._clean_prompt_fragment(text)
        if not cleaned:
            return ""
        cleaned = re.split(r"[.;:]", cleaned, maxsplit=1)[0].strip(" ,.")
        cleaned = re.sub(rf"^{re.escape(char_name)}\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.")
        cleaned = re.sub(r"^(has|is|with|wears?|wearing)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(he|she|they|it)\s+has\s+", "", cleaned, flags=re.IGNORECASE)
        lowered = cleaned.lower()
        color_terms = [
            "black", "brown", "dark brown", "light brown", "blonde", "auburn",
            "red", "gray", "grey", "white", "silver", "green", "blue", "hazel",
        ]
        matched_color = next((term for term in color_terms if term in lowered), "")
        if "hair" in lowered and len(cleaned.split()) > 4:
            if "ponytail" in lowered:
                cleaned = f"{matched_color} ponytail".strip() if matched_color else "ponytail"
            elif "braid" in lowered:
                cleaned = f"{matched_color} braid".strip() if matched_color else "braid"
            else:
                cleaned = f"{matched_color} hair".strip() if matched_color else "hair"
        return self._clean_prompt_fragment(cleaned)

    def _select_wardrobe_terms(
        self,
        wardrobe_terms: List[str],
        multi_character_panel: bool,
        max_terms: Optional[int] = None,
    ) -> List[str]:
        """Prefer color-bearing wardrobe anchors and keep them compact."""
        import re

        color_words = {
            "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
            "brown", "pink", "purple", "orange", "gold", "silver", "beige",
            "tan", "navy", "cream", "burgundy", "forest", "golden",
            "khaki", "olive", "maroon", "teal", "coral", "blonde", "auburn",
            "ginger", "chestnut", "raven", "strawberry",
        }
        identity_only_markers = {
            "young woman", "young man", "adult male", "adult female", "adult woman",
            "adult man", "person", "woman", "man",
        }
        garment_words = {
            "shirt", "blouse", "dress", "coat", "jacket", "hoodie", "sweater",
            "pants", "jeans", "skirt", "boots", "shoes", "heels", "hat",
            "scarf", "gloves", "uniform", "suit", "vest", "onesie",
            "tie", "t-shirt", "shorts", "socks", "robe", "cape", "belt",
            "romper", "overalls", "sneakers", "sandals", "cap", "trousers",
            "cardigan", "blazer", "turtleneck",
        }
        cleaned_terms = []
        for term in wardrobe_terms:
            cleaned = self._clean_prompt_fragment(term)
            cleaned = re.sub(r"^(is|wears?|wearing)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^[A-Z][a-z]+\s+is\s+", "", cleaned)
            cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
            cleaned = self._clean_prompt_fragment(cleaned)
            tokens = {token.strip(" ,.") for token in cleaned.lower().split()}
            if cleaned and tokens and tokens.issubset(color_words):
                continue
            lowered = cleaned.lower()
            if lowered in identity_only_markers or (
                any(marker in lowered for marker in identity_only_markers)
                and not any(garment in lowered for garment in garment_words)
            ):
                continue
            # Reject terms that are neither a color word nor contain a garment word
            if lowered not in color_words and not any(g in lowered for g in garment_words):
                continue
            if cleaned:
                cleaned_terms.append(cleaned)

        def score(term: str) -> Tuple[int, int]:
            tokens = {token.strip(" ,.") for token in term.lower().split()}
            has_color = int(bool(tokens.intersection(color_words)))
            garment_weights = {
                "shirt": 3,
                "blouse": 3,
                "dress": 3,
                "coat": 3,
                "jacket": 3,
                "romper": 2,
                "overalls": 2,
                "hoodie": 2,
                "sweater": 2,
                "boots": 2,
                "sandals": 2,
                "sneakers": 2,
                "pants": 1,
                "jeans": 1,
                "skirt": 1,
                "hat": 1,
                "cap": 1,
                "scarf": 1,
                "gloves": 1,
            }
            garment_score = max(
                (weight for garment, weight in garment_weights.items() if garment in term.lower()),
                default=0,
            )
            return has_color + garment_score, -len(term)

        selected = []
        seen = set()
        for term in sorted(cleaned_terms, key=score, reverse=True):
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            selected.append(term)
            limit = max_terms if max_terms is not None else (1 if multi_character_panel else 2)
            if len(selected) >= limit:
                break
        return selected

    def _select_identity_terms(
        self,
        identity_terms: List[str],
        max_terms: int,
    ) -> List[str]:
        """Prefer face-distinctive identity cues over generic body descriptors."""
        feature_weights = {
            "glasses": 4,
            "hair": 3,
            "eyes": 3,
            "ponytail": 3,
            "braid": 3,
            "beard": 3,
            "mustache": 3,
            "freckles": 2,
            "scar": 2,
            "mechanical": 5,
            "articulated": 5,
            "limb": 4,
            "robot": 1,
            "sensor": 4,
            "circuit": 4,
            "joint": 4,
            "metal": 4,
            "chrome": 4,
            "feather": 4,
            "plumage": 4,
            "fur": 4,
            "coat": 4,
            "pelt": 4,
            "scale": 4,
            "wing": 3,
            "beak": 3,
            "bird": 1,
            "whisker": 2,
            "skin": 1,
            "build": 1,
            "young": 0,
            "adult": 0,
            "male": 0,
            "female": 0,
        }

        color_words = {
            "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
            "brown", "pink", "purple", "orange", "gold", "silver", "beige",
            "tan", "navy", "cream", "burgundy", "forest", "golden",
            "khaki", "olive", "maroon", "teal", "coral", "blonde", "auburn",
            "ginger", "chestnut", "raven", "strawberry",
        }

        def score(term: str) -> Tuple[int, int]:
            lowered = term.lower()
            weight = max((value for token, value in feature_weights.items() if token in lowered), default=0)
            if "hair" in lowered and any(color in lowered for color in color_words):
                weight += 2
            # Color bonus for any term containing a color word (not just hair)
            if any(color in lowered for color in color_words):
                weight += 1
            # Penalty for purely generic single-token terms with no color/pattern info
            generic_markers = {"fur", "coat", "pelt", "hair", "skin", "feathers", "scales", "wool"}
            if lowered.strip() in generic_markers:
                weight -= 2
            return weight, -len(term)

        selected = []
        seen = set()
        for term in sorted(identity_terms, key=score, reverse=True):
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            selected.append(term)
            if len(selected) >= max_terms:
                break
        return selected

    def _select_single_nonhuman_identity_terms(
        self,
        entity_state: Dict[str, Any],
        char_info: Any = None,
    ) -> List[str]:
        """Prefer one rich, color-bearing appearance anchor for single non-human panels."""
        entity_type = str(entity_state.get("entity_type", "")).lower()
        if entity_type not in {"robot", "bird", "mammal", "reptile", "fish"}:
            return []

        color_tokens = {
            "black", "white", "brown", "dark", "light", "blonde", "auburn", "red",
            "gray", "grey", "silver", "green", "blue", "hazel", "golden", "tan",
            "orange", "pink", "cream", "yellow", "gunmetal", "chrome",
        }
        marker_map = {
            "robot": {"metal", "mechanical", "steel", "chrome", "sensor", "led", "circuit", "joint", "chassis"},
            "bird": {"feather", "plumage", "wing", "beak", "talon"},
            "mammal": {"fur", "coat", "pelt", "whisker", "paw", "tail", "ear", "mane", "snout", "stripe", "spot"},
            "reptile": {"scale", "shell", "crest", "hide"},
            "fish": {"scale", "fin", "gill", "aquatic"},
        }
        generic_terms = {
            "fur", "coat", "pelt", "feathers", "plumage", "scales",
            "bird", "mammal", "reptile", "fish", "robot", "dog", "cat",
        }

        raw_terms: List[str] = []
        raw_terms.extend(entity_state.get("face_terms", []) or [])
        if char_info is not None:
            for attr in ("appearance_details", "visual_description"):
                value = getattr(char_info, attr, "")
                if value:
                    raw_terms.extend(str(value).split(","))
            raw_terms.extend(str(item) for item in (getattr(char_info, "key_attributes", None) or []))

        cleaned_terms = []
        seen = set()
        name = str(entity_state.get("name", "")).strip()
        for term in raw_terms:
            cleaned = self._clean_identity_term(term, name)
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            cleaned_terms.append(cleaned)

        def score(term: str) -> Tuple[int, int, int]:
            lowered = term.lower()
            tokens = {token.strip(" ,.") for token in lowered.split()}
            score_value = 0
            if tokens.intersection(color_tokens):
                score_value += 4
            if any(marker in lowered for marker in marker_map.get(entity_type, set())):
                score_value += 3
            if len(tokens) >= 3:
                score_value += 2
            if lowered in generic_terms:
                score_value -= 4
            return score_value, len(tokens), len(term)

        ranked = sorted(cleaned_terms, key=score, reverse=True)
        for term in ranked:
            if term.lower() not in generic_terms:
                return [term]
        return ranked[:1]

    def _collect_character_identity_fallback_terms(
        self,
        char_name: str,
        char_info: Any,
    ) -> List[str]:
        """Mine richer appearance cues from character fields when StoryState terms are too generic."""
        if not char_info:
            return []

        sources = []
        for attr in ("visual_description", "appearance_details"):
            value = getattr(char_info, attr, "")
            if value:
                sources.extend(str(value).split(","))
        key_attributes = getattr(char_info, "key_attributes", None) or []
        sources.extend(str(attr) for attr in key_attributes[:3] if str(attr).strip())

        cleaned_terms = []
        seen = set()
        for source in sources:
            cleaned = self._clean_identity_term(source, char_name)
            lowered = cleaned.lower()
            if not cleaned or lowered == char_name.lower() or lowered in seen:
                continue
            seen.add(lowered)
            cleaned_terms.append(cleaned)
        return cleaned_terms

    def _enrich_identity_terms(
        self,
        char_name: str,
        selected_terms: List[str],
        char_info: Any,
        max_terms: int,
    ) -> List[str]:
        """Replace generic identity fragments like bare 'hair' with richer character-specific cues."""
        if not selected_terms or not char_info:
            return selected_terms

        generic_terms = {"hair", "eyes", "face", "kind eyes"}
        if not any(term.lower() in generic_terms for term in selected_terms):
            return selected_terms

        merged = []
        seen = set()
        for term in selected_terms:
            lowered = term.lower()
            if lowered in generic_terms:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(term)

        for fallback in self._select_identity_terms(
            self._collect_character_identity_fallback_terms(char_name, char_info),
            max_terms,
        ):
            lowered = fallback.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(fallback)
            if len(merged) >= max_terms:
                break

        if not merged:
            return selected_terms
        return merged[:max_terms]

    def _get_cross_panel_link(
        self,
        story_state: Optional[Dict[str, Any]],
        panel_id: int,
    ) -> Dict[str, Any]:
        """Fetch the continuity link targeting the current panel."""
        if not story_state:
            return {}
        for link in story_state.get("cross_panel_links", []):
            if link.get("to_panel") == panel_id:
                return link
        return {}

    def _summarize_scene_anchor(self, setting: str) -> str:
        """Compress verbose setting strings into a stable scene anchor."""
        clauses = [
            self._clean_prompt_fragment(part)
            for part in str(setting or "").split(",")
            if self._clean_prompt_fragment(part)
        ]
        if not clauses:
            return ""
        return self._join_segments_with_budget(clauses[:2], 72)

    def _scene_has_redundant_in_prefix(self, text: str) -> bool:
        import re

        return bool(
            re.match(
                r"^in\s+",
                str(text or "").strip(),
                flags=re.IGNORECASE,
            )
        )

    def _build_scene_text(self, setting: str) -> str:
        scene_anchor = self._summarize_scene_anchor(setting)
        scene_text = self._clean_prompt_fragment(scene_anchor or setting)
        if not scene_text:
            return ""
        if self._scene_has_redundant_in_prefix(scene_text):
            return scene_text
        return self._clean_prompt_fragment(f"in {scene_text}")

    def _build_scene_continuity_text(self, scene: str) -> str:
        import re

        current_scene = self._clean_prompt_fragment(scene)
        if not current_scene:
            return ""
        if self._scene_has_redundant_in_prefix(current_scene):
            stripped = re.sub(
                r"^in\s+",
                "",
                current_scene,
                flags=re.IGNORECASE,
            ).strip(" ,.")
            if stripped:
                return self._clean_prompt_fragment(f"same scene with {stripped}")
        return self._clean_prompt_fragment(f"same {current_scene} background")

    def _build_expression_anchor(
        self,
        panel_state: Dict[str, Any],
        present_char_names: List[str],
    ) -> str:
        """Compile compact per-panel facial expression cues."""
        emotion_cues = panel_state.get("emotion_cues") or {}
        if not emotion_cues:
            return ""

        segments = []
        for char_name in present_char_names[:2]:
            cue = self._clean_prompt_fragment(
                emotion_cues.get(char_name)
                or emotion_cues.get(str(char_name).lower(), "")
            )
            if not cue or cue == "neutral":
                continue
            if len(present_char_names) == 1:
                segments.append(f"{cue} expression")
            else:
                segments.append(f"{char_name} {cue}")

        return self._join_segments_with_budget(segments, 48)

    def _build_panel_entity_anchor(
        self,
        entity_state: Dict[str, Any],
        multi_character_panel: bool,
        char_info: Any = None,
        allow_extra_wardrobe: bool = False,
        single_nonhuman_identity_boost: bool = False,
    ) -> str:
        """Compile one explicit per-entity prompt anchor from StoryState."""
        name = str(entity_state.get("name", "")).strip()
        if not name:
            return ""

        # === Cross-panel wardrobe consistency cache ===
        # Cache only the wardrobe segment (not full anchor) to reuse across panels,
        # preventing wardrobe drift while respecting per-panel format differences
        # (single-character vs multi-character budget, position prefix).
        if not hasattr(self, '_wardrobe_preferences'):
            self._wardrobe_preferences = {}
        cached = self._wardrobe_preferences.get(name.lower())
        if cached is not None:
            # Use cached wardrobe terms instead of selecting fresh ones
            wardrobe_cache_hit = True
            cached_wardrobe = list(cached)
        else:
            wardrobe_cache_hit = False
            cached_wardrobe = None

        segments = [name]
        entity_type = self._clean_prompt_fragment(entity_state.get("entity_type", "human")) or "human"
        position = self._clean_prompt_fragment(entity_state.get("position", ""))
        if multi_character_panel and position:
            segments.append(f"on the {position}")
        gender = self._clean_prompt_fragment(entity_state.get("gender", ""))
        age_bucket = self._clean_prompt_fragment(entity_state.get("age_bucket", ""))
        if entity_type == "human" and age_bucket and age_bucket not in {"adult", "neutral"}:
            segments.append(age_bucket.replace("_", " "))
        if entity_type == "human" and gender and gender not in {"neutral", "unknown"}:
            segments.append(gender)

        face_terms = self._select_identity_terms(
            [
                self._clean_identity_term(term, name)
                for term in entity_state.get("face_terms", [])
                if self._clean_identity_term(term, name)
            ],
            2 if entity_type != "human" else (1 if multi_character_panel else 2),
        )
        face_terms = self._enrich_identity_terms(
            name,
            face_terms,
            char_info,
            2 if entity_type != "human" else (1 if multi_character_panel else 2),
        )
        if entity_type in {"bird", "mammal", "reptile", "fish", "robot"} and not multi_character_panel and single_nonhuman_identity_boost:
            single_nonhuman_terms = self._select_single_nonhuman_identity_terms(
                entity_state,
                char_info=char_info,
            )
            if single_nonhuman_terms:
                face_terms = single_nonhuman_terms
        if entity_type == "human" and not multi_character_panel:
            hair_terms = [
                self._clean_identity_term(term, name)
                for term in entity_state.get("face_terms", [])
                if "hair" in str(term).lower() and self._clean_identity_term(term, name)
            ]
            if hair_terms and not any("hair" in term.lower() for term in face_terms):
                face_terms = [hair_terms[0], *face_terms]
                face_terms = list(dict.fromkeys(face_terms))[:2]
        if face_terms:
            segments.extend(face_terms)

        max_wardrobe_terms = 2 if (entity_type == "human" and (allow_extra_wardrobe or not multi_character_panel)) else 1
        if wardrobe_cache_hit and cached_wardrobe is not None:
            # Reuse cached wardrobe terms for cross-panel consistency
            wardrobe_terms = list(cached_wardrobe)[:max_wardrobe_terms]
        else:
            wardrobe_terms = self._select_wardrobe_terms(
                entity_state.get("wardrobe_terms", []),
                multi_character_panel=multi_character_panel,
                max_terms=max_wardrobe_terms,
            )
            if (
                entity_type == "human"
                and allow_extra_wardrobe
                and char_info is not None
                and len(wardrobe_terms) < 2
            ):
                fallback_wardrobe_terms = self._select_wardrobe_terms(
                    self._split_state_terms(getattr(char_info, "clothing", "")),
                    multi_character_panel=False,
                    max_terms=2,
                )
                for term in fallback_wardrobe_terms:
                    if term.lower() in {item.lower() for item in wardrobe_terms}:
                        continue
                    wardrobe_terms.append(term)
                    if len(wardrobe_terms) >= 2:
                        break
            # Merge palette colors into wardrobe terms unconditionally
            # This ensures clothing has color info even when the parser's clothing field lacks it
            palette = entity_state.get("clothing_palette") or entity_state.get("wardrobe_palette", [])
            if palette and entity_type == "human":
                # Only add colors not already present in wardrobe terms
                all_text = " ".join(wardrobe_terms).lower()
                new_colors = [c for c in palette if c.lower() not in all_text]
                if new_colors:
                    wardrobe_terms = (new_colors + wardrobe_terms)[:max_wardrobe_terms]
            # Cache only the wardrobe terms (not the full anchor) for cross-panel consistency
            wardrobe_key = name.lower()
            if entity_type == "human" and wardrobe_terms and wardrobe_key not in self._wardrobe_preferences:
                self._wardrobe_preferences[wardrobe_key] = tuple(wardrobe_terms)

        if entity_type == "human" and wardrobe_terms:
            segments.extend(wardrobe_terms)

        return self._join_segments_with_budget(
            segments,
            72 if multi_character_panel else 96,
        )

    def _build_layout_from_entities(
        self,
        panel_entities: List[Dict[str, Any]],
    ) -> str:
        """Prefer explicit entity positions over coarse shared layout text."""
        positioned = []
        for entity in panel_entities[:3]:
            position = self._clean_prompt_fragment(entity.get("position", ""))
            name = self._clean_prompt_fragment(entity.get("name", ""))
            if position and name:
                positioned.append(f"{name} {position}")
        return self._join_segments_with_budget(positioned, 56)

    def _build_count_anchor(
        self,
        expected_count: int,
        present_char_names: List[str],
        panel_entities: Optional[List[Dict[str, Any]]] = None,
        compact: bool = False,
    ) -> str:
        """Compile a neutral count anchor that does not assume humans only."""
        if expected_count <= 1:
            return ""
        count_word = {
            2: "two",
            3: "three",
            4: "four",
        }.get(expected_count, str(expected_count))
        entity_types = {
            self._clean_prompt_fragment(entity.get("entity_type", "")).lower()
            for entity in (panel_entities or [])
            if self._clean_prompt_fragment(entity.get("entity_type", ""))
        }
        if compact:
            if entity_types and entity_types.issubset({"human"}):
                return f"{count_word} people"
            if entity_types and entity_types.issubset({"mammal", "bird", "reptile", "fish", "animal"}):
                return f"{count_word} animals"
            return f"{count_word} characters"
        visible_names = " and ".join(present_char_names[:3])
        if visible_names:
            return f"{count_word} characters: {visible_names}"
        return f"{count_word} characters"

    def _get_character_anchor(
        self,
        char_name: str,
        char_info: Any,
        story_state: Optional[Dict[str, Any]],
        multi_character_panel: bool,
    ) -> str:
        """Build a compact identity anchor from StoryState first, then character fields."""
        story_characters = {}
        if story_state:
            story_characters = {
                str(name).lower(): state
                for name, state in story_state.get("character_states", {}).items()
            }

        state = story_characters.get(char_name.lower(), {})
        segments = [char_name]

        identity_terms = [
            self._clean_identity_term(term, char_name)
            for term in state.get("identity_core", [])
            if self._clean_identity_term(term, char_name)
        ]
        identity_terms = [
            term for term in identity_terms
            if term.lower() != char_name.lower()
        ]
        wardrobe_terms = self._select_wardrobe_terms(
            list(state.get("wardrobe_state", [])),
            multi_character_panel=multi_character_panel,
        )
        if identity_terms:
            max_identity_terms = 1 if multi_character_panel else 2
            selected_identity_terms = self._select_identity_terms(identity_terms, max_identity_terms)
            selected_identity_terms = self._enrich_identity_terms(
                char_name,
                selected_identity_terms,
                char_info,
                max_identity_terms,
            )
            segments.extend(selected_identity_terms)
        if wardrobe_terms:
            segments.extend(wardrobe_terms)

        if len(segments) == 1:
            appearance_sources = []
            if hasattr(char_info, "appearance_details") and char_info.appearance_details:
                appearance_sources.extend(str(char_info.appearance_details).split(","))
            if hasattr(char_info, "visual_description") and char_info.visual_description:
                appearance_sources.extend(str(char_info.visual_description).split(","))
            for source in appearance_sources:
                cleaned = self._clean_identity_term(source, char_name)
                if cleaned and cleaned.lower() != char_name.lower():
                    segments.append(cleaned)
                if len(segments) >= (3 if multi_character_panel else 4):
                    break

            if hasattr(char_info, "clothing") and char_info.clothing:
                clothing_terms = self._select_wardrobe_terms(
                    str(char_info.clothing).split(","),
                    multi_character_panel=multi_character_panel,
                )
                segments.extend(clothing_terms)

        if len(segments) == 1 and hasattr(char_info, "key_attributes") and char_info.key_attributes:
            segments.extend(
                self._clean_prompt_fragment(str(attr))
                for attr in char_info.key_attributes[:2]
                if self._clean_prompt_fragment(str(attr))
            )

        return self._join_segments_with_budget(
            segments,
            64 if multi_character_panel else 96,
        )

    def _compose_prompt(
        self,
        panel: Panel,
        global_style: str,
        characters: Dict,
        panel_index: int = 0,
        all_panels: List[Panel] = None,
        consistency_constraints: List[str] = None,
        story_state: Optional[Dict[str, Any]] = None,
        return_plan: bool = False,
    ) -> str:
        """
        Compose final generation prompt optimized for SDXL.
        Structure: Character (START) > Scene Description > Key Objects > Time > Style
        
        CRITICAL: Each panel MUST have character description. If LLM output is incomplete,
        we fall back to character data or scene content.
        """
        import re

        def clean_identity_fragment(fragment: str, char_name: str, max_words: int) -> str:
            fragment = str(fragment).strip(" ,.")
            if not fragment:
                return ""
            fragment = re.sub(
                rf"^{re.escape(char_name)}\s+(has|is)\s+",
                "",
                fragment,
                flags=re.IGNORECASE,
            )
            fragment = re.sub(r"^(with|wearing)\s+", "", fragment, flags=re.IGNORECASE)
            words = fragment.split()
            fragment = " ".join(words[:max_words]).strip(" ,.")
            if len(fragment) <= 2:
                return ""
            return fragment

        def extract_raw_action_fallback(raw_text: str) -> str:
            raw_text = re.sub(r"<[^>]+>", "", raw_text or "").strip()
            match = re.search(
                r"\b(walks?|runs?|sits?|stands?|looks?|waits?|pauses?|gets?|drives?|writes?|reads?|"
                r"eats?|smiles?|talks?|continues?|boards?|holds?|stretches?|crosses?|stops?|scans?|"
                r"flies?|flys?|pecks?|perches?|lands?|sorts?|places?|moves?|inspects?|surveys?)\b([^.;,]*)",
                raw_text,
                flags=re.IGNORECASE,
            )
            if not match:
                return ""
            phrase = f"{match.group(1)}{match.group(2)}".strip(" ,.")
            phrase = re.sub(r"^(he|she|they|it)\s+", "", phrase, flags=re.IGNORECASE)
            phrase = re.sub(r"\s+", " ", phrase)
            return phrase

        def action_keywords(text: str) -> List[str]:
            return [
                token
                for token in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(token) >= 3 and token not in {
                    "the", "and", "with", "into", "onto", "from", "near", "over",
                    "under", "while", "that", "this", "those", "these", "then",
                }
            ]

        def has_distinctive_action_detail(text: str) -> bool:
            detail_markers = {
                "notes", "note", "hand", "hands", "leaf", "leaves", "book", "bag",
                "keyboard", "screen", "gloves", "ball", "ticket", "map",
            }
            lowered = (text or "").lower()
            return any(marker in lowered for marker in detail_markers)

        panel_state = self._get_story_panel_state(story_state, panel.panel_id)
        cross_panel_link = self._get_cross_panel_link(story_state, panel.panel_id)
        entity_prompt_contract = bool(self.config.get("enable_entity_prompt_contract", True))
        compiler_annotations: List[str] = []

        # === STEP 1: Extract character information ===
        character_lookup = {
            str(name).lower(): name
            for name in characters.keys()
        }
        story_state_characters = []
        panel_entities = [dict(entity) for entity in (panel_state.get("panel_entities") or [])]
        for name in panel_state.get("characters_present", []):
            resolved_name = character_lookup.get(str(name).lower())
            if resolved_name and resolved_name not in story_state_characters:
                story_state_characters.append(resolved_name)
        present_char_names = story_state_characters or self._extract_characters_from_panel(panel, characters)
        expected_count = int(panel_state.get("expected_count") or len(present_char_names))
        if (
            self.config.get("enable_role_bound_multi_character_anchors", True)
            and entity_prompt_contract
            and len(panel_entities) == 2
            and all(str(entity.get("entity_type", "")).lower() == "human" for entity in panel_entities)
            and not any(self._clean_prompt_fragment(entity.get("position", "")) for entity in panel_entities)
        ):
            layout_hint = str(panel_state.get("spatial_layout", "") or "").lower()
            if "left/right" in layout_hint or "stable left/right" in layout_hint:
                panel_entities = [
                    {**entity, "position": "left" if index == 0 else "right"}
                    for index, entity in enumerate(panel_entities)
                ]
                compiler_annotations.append("multi_human_position_bound")
        
        # CRITICAL FIX: DO NOT hardcode character count!
        # Let LLM flexibility handle stories with "meet his friends" or other multi-person scenarios
        # Only use character count hint if we have explicit information
        
        # Build full character description from character data
        # CRITICAL FIX: Use visual_description as SINGLE SOURCE OF TRUTH
        # Key_attributes and clothing may contradict visual_description
        char_descriptions = []
        multi_character_panel = len(present_char_names) > 1
        if entity_prompt_contract and panel_entities:
            allow_extra_wardrobe = expected_count <= 2
            single_nonhuman_identity_boost = (
                len(panel_entities) == 1
                and str(panel_entities[0].get("entity_type", "")).lower() in {"bird", "mammal", "reptile", "fish", "robot"}
            )
            for entity_state in panel_entities:
                entity_name = str(entity_state.get("name", "")).lower()
                resolved_name = character_lookup.get(entity_name)
                desc = self._build_panel_entity_anchor(
                    entity_state,
                    multi_character_panel,
                    allow_extra_wardrobe=allow_extra_wardrobe,
                    single_nonhuman_identity_boost=single_nonhuman_identity_boost,
                )
                if resolved_name and resolved_name in characters:
                    desc = self._build_panel_entity_anchor(
                        entity_state,
                        multi_character_panel,
                        char_info=characters[resolved_name],
                        allow_extra_wardrobe=allow_extra_wardrobe,
                        single_nonhuman_identity_boost=single_nonhuman_identity_boost,
                    )
                if desc:
                    char_descriptions.append(desc)
        else:
            for char_name in present_char_names:
                if char_name in characters:
                    char = characters[char_name]
                    desc = self._get_character_anchor(
                        char_name=char_name,
                        char_info=char,
                        story_state=story_state,
                        multi_character_panel=multi_character_panel,
                    )
                    if desc:
                        char_descriptions.append(desc)
        
        # === STEP 2: Build scene description ===
        scene_desc = ""
        raw_scene = panel.raw_prompt or panel.enhanced_prompt or ""
        if entity_prompt_contract:
            scene_desc = re.sub(r'<[^>]+>\s*', '', raw_scene).strip()
            for char_name in present_char_names:
                scene_desc = re.sub(rf"\b{re.escape(char_name)}\b", "", scene_desc, count=1, flags=re.IGNORECASE)
            scene_desc = re.sub(r"\s+", " ", scene_desc).strip(" ,.")
            if not scene_desc and panel.enhanced_prompt:
                scene_desc = re.sub(r'<[^>]+>\s*', '', panel.enhanced_prompt).strip()
        elif panel.enhanced_prompt and len(panel.enhanced_prompt) > 15:
            scene_desc = re.sub(r'<[^>]+>\s*', '', panel.enhanced_prompt).strip()
        else:
            scene_desc = re.sub(r'<[^>]+>\s*', '', panel.raw_prompt).strip()
        
        # Remove photorealistic/quality terms from scene_desc (we add them later)
        quality_terms = ["photorealistic", "realistic photography", "sharp focus", 
                        "8k detailed", "highly detailed", "masterpiece"]
        for term in quality_terms:
            scene_desc = re.sub(rf',\s*{re.escape(term)}', '', scene_desc, flags=re.IGNORECASE)
            scene_desc = re.sub(rf'{re.escape(term)}\s*,\s*', '', scene_desc, flags=re.IGNORECASE)
        scene_desc = scene_desc.rstrip(',. ')
        
        # === STEP 2b: Object Propagation across panels ===
        # If this isn't the first panel, check for objects from previous panels
        # that give context to generic terms (door→bus door, window→car window)
        if panel_index > 0 and all_panels:
            # Collect all raw prompts from previous panels
            prev_context = " ".join(
                re.sub(r'<[^>]+>', '', p.raw_prompt).strip()
                for p in all_panels[:panel_index]
            )
            # Extract key nouns (potential objects) from previous context
            prev_nouns = set()
            for word in prev_context.lower().split():
                word = word.strip('.,!?;:\'"()[]{}')
                if word in ['bus', 'train', 'car', 'taxi', 'airport', 'station', 'park',
                           'cafe', 'restaurant', 'shop', 'school', 'office', 'factory',
                           'kitchen', 'bedroom', 'garden', 'bridge', 'street', 'road']:
                    prev_nouns.add(word)
            
            # If we have context objects, enhance generic terms in scene_desc
            if prev_nouns:
                # Use the most recently introduced object (last in the set from context order)
                ctx_word = sorted(prev_nouns, key=lambda x: -prev_context.lower().split().index(x) if x in prev_context.lower().split() else 0)[0]
                generic_terms = ['door', 'window', 'inside', 'interior', 'entrance', 'seat']
                for generic in generic_terms:
                    if generic in scene_desc.lower():
                        has_context = any(noun in scene_desc.lower() for noun in prev_nouns)
                        if not has_context:
                            scene_desc = re.sub(rf'\b{generic}\b', f'{ctx_word} {generic}', scene_desc, flags=re.IGNORECASE)

        state_targets = []
        continuity_targets = []
        scene_continuity_text = ""
        scene_continuity_required = False
        camera_text = ""
        action_text = scene_desc
        raw_action_fallback = ""
        panel_props = []
        expression_text = self._build_expression_anchor(panel_state, present_char_names)
        if panel_state:
            for continuity_item in panel_state.get("continuity_from_prev", []):
                lowered = continuity_item.lower()
                if lowered.startswith("carry over "):
                    carried_items = [
                        self._clean_prompt_fragment(piece)
                        for piece in continuity_item.replace("carry over ", "", 1).split(",")
                    ]
                    continuity_targets.extend([item for item in carried_items if item])
                elif lowered.startswith("remain in "):
                    same_scene = continuity_item.replace("remain in ", "", 1).strip()
                    if same_scene:
                        scene_continuity_text = self._build_scene_continuity_text(
                            self._summarize_scene_anchor(same_scene)
                        )
                # NEW: Handle explicit scene transitions
                elif lowered.startswith("transition from "):
                    # Extract the target scene after "transition from X to Y"
                    transition_match = re.match(r"transition from .+ to (.+)", lowered)
                    if transition_match:
                        target_scene = transition_match.group(1).strip()
                        if target_scene:
                            scene_continuity_text = self._build_scene_continuity_text(
                                self._summarize_scene_anchor(target_scene)
                            )
            panel_characters = {str(name).lower() for name in panel_state.get("characters_present", [])}
            panel_actions = {str(item).lower() for item in panel_state.get("action_beats", [])}
            panel_setting = self._normalize_story_label(panel.setting)
            panel_layout = self._normalize_story_label(panel_state.get("spatial_layout", ""))
            for item in panel_state.get("must_show", []):
                if item not in state_targets:
                    state_targets.append(item)
                lowered = str(item).lower()
                if lowered in panel_characters or lowered in panel_actions:
                    continue
                if self._normalize_story_label(item) == panel_setting or self._normalize_story_label(item) == panel_layout:
                    continue
                if item not in panel_props:
                    panel_props.append(item)
            action_sources = panel_state.get("action_beat_sources", {})
            if panel_state.get("action_beats"):
                primary_state_action = str(panel_state["action_beats"][0] or "")
                if action_sources.get(primary_state_action) == "raw_prompt":
                    raw_action_fallback = self._clean_prompt_fragment(primary_state_action)
            if not raw_action_fallback:
                raw_action_fallback = extract_raw_action_fallback(panel.raw_prompt)
            if panel_state.get("action_beats"):
                filtered_actions = []
                for beat in panel_state["action_beats"]:
                    raw_beat = str(beat or "").strip().lower()
                    cleaned_beat = self._clean_prompt_fragment(beat)
                    if not cleaned_beat:
                        continue
                    if raw_beat in {"smiles", "smiling", "looks"} and expression_text:
                        continue
                    filtered_actions.append(cleaned_beat)
                action_text = (filtered_actions[0] if filtered_actions else "") or raw_action_fallback or scene_desc
            else:
                action_text = raw_action_fallback or scene_desc
        else:
            raw_action_fallback = extract_raw_action_fallback(panel.raw_prompt)

        primary_entity_type = ""
        if panel_entities:
            primary_entity_type = str(panel_entities[0].get("entity_type", "")).lower()
        if primary_entity_type in {"robot", "bird", "mammal", "reptile", "fish"} and raw_action_fallback:
            scene_like_tokens = {
                "sunlight", "lighting", "shadow", "leaves", "sky", "trees", "forest",
                "park", "background", "undergrowth", "feathers", "cloud", "grass",
            }
            action_lower = action_text.lower()
            if action_lower == scene_desc.lower() or any(token in action_lower for token in scene_like_tokens):
                action_text = raw_action_fallback
        if primary_entity_type == "human" and raw_action_fallback:
            raw_keywords = action_keywords(raw_action_fallback)
            action_keywords_current = action_keywords(action_text)
            raw_only_keywords = [token for token in raw_keywords if token not in action_keywords_current]
            keyword_overlap = {
                token for token in action_keywords_current
                if token in raw_keywords
            }
            generic_human_actions = {
                "looking ahead", "slowing down", "stopping", "focused expression",
                "relaxed posture", "interacts with counter staff", "checks rearview mirror",
                "looking at nina",
            }
            preserve_specific_action = (
                has_distinctive_action_detail(action_text)
                and not has_distinctive_action_detail(raw_action_fallback)
            )
            if (
                action_text.lower() in generic_human_actions
                or (
                    not preserve_specific_action
                    and (
                        (len(raw_only_keywords) >= 2 and len(raw_keywords) > len(action_keywords_current))
                        or (action_keywords_current and len(keyword_overlap) < max(1, len(action_keywords_current) // 2))
                    )
                )
            ):
                action_text = raw_action_fallback
        continuity_targets = self._extend_travel_continuity_targets(
            story_state=story_state,
            panel_state=panel_state,
            panel=panel,
            action_text=action_text,
            continuity_targets=continuity_targets,
            panel_props=panel_props,
            primary_entity_type=primary_entity_type,
        )
        visible_props = {str(item).lower() for item in (panel_props + continuity_targets)}
            # Entity-type-specific action rewriting removed for generalization.
            # All entity types use the generic action text.

        state_targets_lower = {str(item).lower() for item in state_targets}
        if cross_panel_link.get("carry_over_props"):
            continuity_targets.extend(
                self._clean_prompt_fragment(prop)
                for prop in cross_panel_link.get("carry_over_props", [])
                if self._clean_prompt_fragment(prop)
            )
        if not scene_continuity_text and cross_panel_link.get("same_scene_segment"):
            current_scene = self._summarize_scene_anchor(getattr(panel, "setting", ""))
            if current_scene:
                scene_continuity_text = self._build_scene_continuity_text(current_scene)

        if getattr(panel, "key_objects", ""):
            for item in [part.strip() for part in str(panel.key_objects).split(",") if part.strip()]:
                if item not in panel_props:
                    panel_props.append(item)

        lower_panel_props = {str(prop).lower() for prop in panel_props}
        deduped_continuity_targets = []
        # window/seat/bench/screen are specific structural elements, not generic across scenes.
        # Only door/chair/table are truly generic enough to suppress.
        generic_continuity_props = {"door", "chair", "table"}
        for item in continuity_targets:
            lowered = str(item).lower()
            if lowered in lower_panel_props:
                continue
            if lowered in {str(existing).lower() for existing in deduped_continuity_targets}:
                continue
            # Never drop state-mandated props (must_show overrides generic suppression)
            if lowered in state_targets_lower:
                deduped_continuity_targets.append(item)
                continue
            if lowered in generic_continuity_props and lowered not in (panel.raw_prompt or "").lower() and lowered not in state_targets_lower:
                continue
            deduped_continuity_targets.append(item)
        continuity_targets = deduped_continuity_targets

        raw_prompt_lower = (panel.raw_prompt or "").lower()
        relation_text = ""
        if "at the door" in raw_prompt_lower and any("door" in str(prop).lower() for prop in panel_props):
            if any("bus door" in str(prop).lower() for prop in panel_props):
                relation_text = "at the bus door"
            elif any("train door" in str(prop).lower() for prop in panel_props):
                relation_text = "at the train door"
            else:
                relation_text = "at the door"
        elif "by the window" in raw_prompt_lower or "beside the window" in raw_prompt_lower:
            if any("car" in str(prop).lower() for prop in panel_props):
                relation_text = "by the car window"
            else:
                relation_text = "by the window"
        elif "book beside her" in raw_prompt_lower or "book beside him" in raw_prompt_lower:
            relation_text = "book beside her"
        elif "in front of" in raw_prompt_lower:
            match = re.search(r"\bin front of\s+(?:a|an|the)?\s*([^.,;]+)", panel.raw_prompt or "", flags=re.IGNORECASE)
            if match:
                target = self._clean_prompt_fragment(match.group(1))
                if target:
                    relation_text = f"in front of {target}"
        elif any("window" in str(prop).lower() for prop in panel_props) and any("car" in str(prop).lower() for prop in panel_props):
            relation_text = "by the car window"

        scene_text = ""
        if getattr(panel, "setting", ""):
            scene_text = self._build_scene_text(panel.setting)
        if relation_text == "at the bus door" and "inside" in scene_text.lower():
            scene_text = "at the bus doorway with the bus interior visible"
        if self.config.get("enable_vehicle_interior_prompting", True):
            vehicle_context = " ".join([
                scene_text,
                scene_desc,
                " ".join(str(prop) for prop in panel_props),
                panel.raw_prompt or "",
            ]).lower()
            action_lower = action_text.lower()
            car_context = "car" in vehicle_context and "bus" not in vehicle_context and "train" not in vehicle_context
            if car_context and any(token in action_lower for token in ("drives", "driving", "drive along")):
                action_text = "driving inside the car"
                relation_text = "in the driver's seat with hands on the steering wheel"
                compiler_annotations.append("vehicle_interior_injected")
            elif car_context and ("scenery" in action_lower or "outside" in (panel.raw_prompt or "").lower()):
                action_text = "looking at the scenery outside from inside the car"
                relation_text = "by the car window in the driver's seat"
                compiler_annotations.append("vehicle_interior_injected")
            elif car_context and "take a break" in action_lower and ("parked" in vehicle_context or "side of road" in vehicle_context):
                relation_text = "beside the parked car"
                compiler_annotations.append("vehicle_break_relation")
        relation_text, support_relation_annotation = self._build_action_support_relation(
            panel=panel,
            panel_state=panel_state,
            action_text=action_text,
            scene_text=scene_text,
            relation_text=relation_text,
            panel_props=panel_props,
            continuity_targets=continuity_targets,
            primary_entity_type=primary_entity_type,
        )
        if support_relation_annotation:
            compiler_annotations.append(support_relation_annotation)

        shot_map = {
            "extreme_closeup": "extreme close-up",
            "closeup": "close-up portrait",
            "medium": "medium shot",
            "wide": "wide shot",
            "over_shoulder": "over-the-shoulder shot",
            "establishing": "establishing shot",
        }
        camera_parts = []
        if panel.shot_type in shot_map:
            camera_parts.append(shot_map[panel.shot_type])
        if panel.camera_movement and panel.camera_movement != "static":
            camera_parts.append(panel.camera_movement.replace("_", " "))
        if panel.lighting_mood and panel.lighting_mood not in ("natural", "", panel.time_of_day):
            camera_parts.append(panel.lighting_mood)
        camera_text = self._clean_prompt_fragment(", ".join(camera_parts))

        identity_slots = [
            {"name": "identity_anchor", "text": desc, "priority": 100 - index, "required": True}
            for index, desc in enumerate(char_descriptions)
        ]

        multi_character_panel = len(present_char_names) > 1
        binding_priority_panel = self._needs_binding_priority_panel(
            panel=panel,
            panel_state=panel_state,
            panel_entities=panel_entities,
            expected_count=expected_count,
        )
        count_text = self._build_count_anchor(
            expected_count,
            present_char_names,
            panel_entities,
            compact=binding_priority_panel,
        )

        layout_text = ""
        if entity_prompt_contract and panel_entities and multi_character_panel:
            layout_text = self._build_layout_from_entities(panel_entities)
        if not layout_text and panel_state.get("spatial_layout") and multi_character_panel:
            if "left/right" in panel_state["spatial_layout"] or "stable left/right" in panel_state["spatial_layout"]:
                layout_text = f"{present_char_names[0]} left, {present_char_names[1]} right"
            else:
                layout_text = panel_state["spatial_layout"]

        prop_text = ""
        if panel_props:
            unique_props = []
            seen_props = set()
            generic_scene_props = {"window", "door", "chair", "table", "seat", "bench", "screen"}
            for prop in panel_props:
                cleaned_prop = self._clean_prompt_fragment(str(prop))
                if not cleaned_prop:
                    continue
                lowered = cleaned_prop.lower()
                if lowered in seen_props:
                    continue
                if self._text_has_term(scene_text, cleaned_prop):
                    continue
                if relation_text and self._text_has_term(relation_text, cleaned_prop):
                    continue
                if self._text_has_term(action_text, cleaned_prop):
                    continue
                if lowered in generic_scene_props and lowered not in raw_prompt_lower and not relation_text and lowered not in state_targets_lower:
                    continue
                seen_props.add(lowered)
                unique_props.append(cleaned_prop)
            if self.config.get("enable_terminal_dish_prompting", True):
                action_lower = action_text.lower()
                if "serve" in action_lower or "finished dish" in action_lower:
                    plated_props = []
                    plated_tokens = ("prepared food", "finished dish", "plated", "bowl", "dish", "plate")
                    prep_tokens = {"vegetables", "cutting board", "chef's knife", "knife"}
                    seen_plated = set()
                    for source_prop in [*continuity_targets, *panel_props]:
                        cleaned_prop = self._clean_prompt_fragment(str(source_prop))
                        lowered = cleaned_prop.lower()
                        if not cleaned_prop or lowered in seen_plated:
                            continue
                        if any(token in lowered for token in plated_tokens):
                            plated_props.append(cleaned_prop)
                            seen_plated.add(lowered)
                    if not plated_props:
                        plated_props.append("plated finished dish")
                    remaining_props = []
                    for prop in unique_props:
                        lowered = prop.lower()
                        if lowered in prep_tokens:
                            continue
                        if lowered in seen_plated:
                            continue
                        remaining_props.append(prop)
                    unique_props = plated_props[:1] + remaining_props
                    compiler_annotations.append("terminal_dish_prioritized")
            prop_text = f"with {', '.join(unique_props[:2])}"

        continuity_focus = []
        seen_focus = set()
        for item in continuity_targets:
            cleaned_item = self._clean_prompt_fragment(str(item))
            lowered = cleaned_item.lower()
            if not cleaned_item or lowered in seen_focus:
                continue
            if (
                self._text_has_term(scene_text, cleaned_item)
                or self._text_has_term(action_text, cleaned_item)
                or self._text_has_term(prop_text, cleaned_item)
            ):
                continue
            if relation_text and self._text_has_term(relation_text, cleaned_item):
                continue
            seen_focus.add(lowered)
            continuity_focus.append(cleaned_item)

        scene_continuity_parts = []
        if scene_continuity_text:
            scene_continuity_parts.append(scene_continuity_text.replace(" background", ""))
        if continuity_focus:
            keep_text = ", ".join(continuity_focus[:2])
            scene_continuity_parts.append(f"keep {keep_text} visible")
        if scene_continuity_parts:
            scene_continuity_text = self._clean_prompt_fragment(", ".join(scene_continuity_parts))
            scene_continuity_required = bool(
                continuity_focus
                and len(present_char_names) <= 1
                and scene_continuity_text
            )

        slots = []
        slots.extend(identity_slots)
        if count_text:
            slots.append({"name": "count_anchor", "text": count_text, "priority": 97, "required": True})
        slots.append({"name": "action_anchor", "text": self._clean_prompt_fragment(action_text), "priority": 92, "required": True})
        if expression_text:
            slots.append({"name": "expression_anchor", "text": expression_text, "priority": 90, "required": False})
        slots.append({
            "name": "scene_anchor",
            "text": scene_text,
            "priority": 88,
            "required": bool(getattr(panel, "setting", "")),
        })
        if scene_continuity_text:
            slots.append({
                "name": "scene_continuity_anchor",
                "text": scene_continuity_text,
                "priority": 87,
                "required": scene_continuity_required,
            })
        if layout_text:
            slots.append({"name": "layout_anchor", "text": self._clean_prompt_fragment(layout_text), "priority": 86, "required": True})
        if relation_text:
            slots.append({"name": "relation_anchor", "text": self._clean_prompt_fragment(relation_text), "priority": 85, "required": True})
        if prop_text:
            slots.append({"name": "prop_anchor", "text": prop_text, "priority": 82, "required": bool(relation_text or panel_props)})
        continuity_text = ", ".join(continuity_targets[:2])
        if continuity_text:
            slots.append({"name": "continuity_anchor", "text": continuity_text, "priority": 72, "required": False})
        camera_required = bool(binding_priority_panel and camera_text)
        if camera_text:
            slots.append({
                "name": "camera_anchor",
                "text": camera_text,
                "priority": 89 if camera_required else 68,
                "required": camera_required,
            })
        time_anchor = self._clean_prompt_fragment(str(panel.time_of_day or ""))
        if time_anchor and time_anchor.lower() not in {"none", "unknown", "unspecified"}:
            slots.append({"name": "time_anchor", "text": time_anchor, "priority": 50, "required": False})
        slots.append({
            "name": "style_anchor",
            "text": "photorealistic, realistic photography, sharp focus, 8k detailed",
            "priority": 0,
            "required": True,
        })

        prompt_char_limit = self._get_prompt_char_limit(binding_priority_panel)
        compiled_prompt, used_slots = self._compile_prompt_from_slots(slots, max_len=prompt_char_limit)
        render_plan = {
            "panel_id": panel.panel_id,
            "prompt_slots": used_slots,
            "compiled_prompt": compiled_prompt,
            "state_targets": panel_state.get("must_show", []),
            "local_constraints": panel_state.get("local_constraints", []),
            "scene_segment": panel_state.get("scene_segment"),
            "expression_targets": panel_state.get("emotion_cues", {}),
            "compiler_annotations": compiler_annotations,
        }

        if return_plan:
            return render_plan
        return compiled_prompt
    
    def _build_prompt_from_components(
        self,
        panel: Panel,
        global_style: str,
        characters: Dict
    ) -> str:
        """Build prompt from individual components (fallback)"""
        import re
        parts = []
        seen_parts = set()

        def add_unique(part: str):
            part_lower = part.lower().strip()
            if part_lower and part_lower not in seen_parts:
                seen_parts.add(part_lower)
                parts.append(part.strip())

        # Get characters in panel
        present_char_names = []
        for char_name, char_info in characters.items():
            if char_name in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                present_char_names.append((char_name, char_info))

        # Character descriptions
        if present_char_names:
            if len(present_char_names) == 2:
                add_unique("two young adults")
            elif len(present_char_names) == 1:
                add_unique("one person")
            
            for char_name, char_info in present_char_names:
                if hasattr(char_info, 'visual_description') and char_info.visual_description:
                    desc_parts = char_info.visual_description.split(",")
                    for desc in desc_parts[:3]:
                        add_unique(desc.strip())
                
                if hasattr(char_info, 'clothing') and char_info.clothing:
                    clothing = char_info.clothing.strip()
                    if len(clothing) > 5:
                        add_unique(clothing)

        # Setting
        if panel.setting:
            add_unique(panel.setting)
        else:
            raw_lower = panel.raw_prompt.lower()
            if "park" in raw_lower:
                add_unique("outdoor park with trees and green grass")
            elif "cafe" in raw_lower or "coffee" in raw_lower:
                add_unique("cozy cafe interior with wooden tables")
            elif "window" in raw_lower:
                add_unique("indoor room with window view")
            elif "exhibition" in raw_lower or "gallery" in raw_lower:
                add_unique("art gallery with paintings on walls")
            elif "bus" in raw_lower:
                add_unique("bus interior")
            elif "train" in raw_lower:
                add_unique("train interior")

        # Main action (cleaned)
        main_content = panel.raw_prompt
        main_content = re.sub(r'<[A-Z][a-z]+>\s*', '', main_content)  # Remove <Name> tags
        main_content = main_content.strip().rstrip('.')
        if main_content:
            add_unique(main_content)

        # Shot type
        shot_map = {
            "closeup": "close-up portrait",
            "medium": "medium shot",
            "wide": "wide angle shot",
            "extreme_closeup": "extreme close-up",
            "over_shoulder": "over-the-shoulder shot"
        }
        if panel.shot_type in shot_map:
            add_unique(shot_map[panel.shot_type])

        # Lighting
        if panel.lighting_mood and panel.lighting_mood != "natural":
            add_unique(f"{panel.lighting_mood} lighting")
        if panel.time_of_day:
            add_unique(panel.time_of_day)

        # Style and quality
        add_unique(global_style)
        add_unique("photorealistic, 8k, sharp focus, cinematic lighting")

        return ", ".join(filter(None, parts))

    def _extract_characters_from_panel(self, panel: Panel, all_characters: Dict) -> List[str]:
        """
        Extract character names appearing in this panel.
        
        CRITICAL FIX: For multi-panel stories, we need to infer character presence:
        - Single-character stories: Character is present in ALL panels
        - Multi-character stories: Check pronouns (they/she/he) to determine count
        """
        import re
        
        present_chars = []
        
        # First, check explicit name mentions in raw_prompt
        for char_name in all_characters.keys():
            if f"<{char_name}>" in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                present_chars.append(char_name)
        
        # CRITICAL FIX: If no explicit mentions found, check story-level heuristics
        if not present_chars:
            all_names = list(all_characters.keys())
            num_chars = len(all_names)
            
            # Single-character story: character is in ALL panels
            if num_chars == 1:
                present_chars = all_names
            
            # Multi-character story: check pronouns
            raw_lower = panel.raw_prompt.lower()
            if 'they' in raw_lower or 'they\'re' in raw_lower:
                # All characters present
                present_chars = all_names
            elif 'she' in raw_lower or 'he' in raw_lower or 'her ' in raw_lower or 'his ' in raw_lower:
                # One character (but which one?) - try to infer from context
                # For simplicity, use first character
                if all_names:
                    present_chars = [all_names[0]]
        
        return present_chars

    @torch.inference_mode()
    def generate_story(
        self,
        production_board: ProductionBoard,
        seed: Optional[int] = None,
        return_portraits: bool = False
    ) -> Tuple[List[Image.Image], Optional[Dict]]:
        """
        Generate complete story as a sequence of images

        Args:
            production_board: LLM-produced production blueprint
            seed: Random seed for reproducibility
            return_portraits: Whether to return character portraits

        Returns:
            Tuple of (generated images list, optional portrait dict)
        """
        if not self._initialized:
            _ = self.base_pipe
            _ = self.portrait_gen

        # Set seed for reproducibility
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[Generate] Starting story: {production_board.story_id}")
        print(f"[Generate] Total frames: {len(production_board.panels)}, Seed: {seed}")

        all_images = []
        portraits = {}

        # Phase 1: Generate character portraits for feature extraction
        print("\n[Generate] Phase 1: Character Portrait Generation...")
        char_dict = {k: v.__dict__ for k, v in production_board.characters.items()}

        try:
            portraits = self.portrait_gen.generate_all_portraits(
                characters=char_dict,
                global_style=production_board.global_style,
                output_dir=f"outputs/portraits/{production_board.story_id}"
            )
            print(f"[Generate] Generated {len(portraits)} character portraits")
        except Exception as e:
            print(f"[Generate] Warning: Portrait generation failed: {e}")
            print("[Generate] Continuing without character portraits...")

        # Phase 2: Batch frame generation (SCA requires batch mode)
        print("\n[Generate] Phase 2: Batch Frame Generation...")

        gen_params = self.config.get("generation_params", {
            "num_steps": 35,
            "guidance_scale": 7.5
        })

        height = self.config.get("height", 1024)
        width = self.config.get("width", 1024)
        num_frames = len(production_board.panels)

        # Step 2a: Compose all prompts first
        prompts = []
        render_plan = []
        identity_images: List[Image.Image] = []
        identity_scales: List[float] = []
        eligible_story_character = self._get_story_face_lock_character(production_board, portraits)
        for i, panel in enumerate(production_board.panels):
            panel_plan = self._compose_prompt(
                panel=panel,
                global_style=production_board.global_style,
                characters=production_board.characters,
                panel_index=i,
                all_panels=production_board.panels,
                consistency_constraints=production_board.consistency_constraints,
                story_state=production_board.story_state,
                return_plan=True,
            )
            prompt = panel_plan["compiled_prompt"]
            prompts.append(prompt)
            render_plan.append(panel_plan)
            reference_image, reference_scale, reference_name = self._get_panel_identity_reference(
                panel=panel,
                panel_state=self._get_story_panel_state(production_board.story_state, panel.panel_id),
                production_board=production_board,
                portraits=portraits,
                eligible_story_character=eligible_story_character,
            )
            if reference_image is None:
                reference_image = Image.new("RGB", (768, 768), color=(127, 127, 127))
            identity_images.append(reference_image)
            identity_scales.append(reference_scale)
            panel_plan["identity_reference"] = reference_name
            panel_plan["identity_reference_scale"] = round(reference_scale, 3)
            print(f"[Frame {i+1}/{num_frames}] Prompt: {prompt[:200]}...")
        production_board.render_plan = render_plan
        story_uses_face_lock = bool(self.config.get("enable_face_lock", False) and max(identity_scales, default=0.0) > 0)

        if not self._initialized:
            if story_uses_face_lock:
                self._ensure_identity_adapter()
            _ = self.attn_processor
            self._configure_runtime_optimizations()
            _ = self.memory_bank
            self._initialized = True
            print("[Pipeline] All components initialized successfully!\n")

        # Clear memory banks for new story (Bug 4 fix: ensure fresh start)
        if self._attn_processor is not None:
            if isinstance(self._attn_processor, dict):
                for proc in self._attn_processor.values():
                    if isinstance(proc, ConsistentSelfAttentionProcessor):
                        proc.clear_memory()
            else:
                self._attn_processor.clear_memory()
        if self._memory_bank is not None:
            self._memory_bank.clear()
        # Clear cross-panel wardrobe cache for fresh story
        self._wardrobe_preferences = {}

        self._set_sca_story_context(production_board.story_state)
        ip_adapter_image_embeds = None
        if story_uses_face_lock:
            ip_adapter_image_embeds = self._build_batch_ip_adapter_embeds(identity_images, identity_scales)

        # Enhanced negative prompt (shared across all frames)
        negative_prompt = (
            "blurry, blurry hands, blurry face, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, fused fingers, too many fingers, "
            "missing fingers, extra fingers, poorly drawn hands, poorly drawn face, "
            "extra people, missing people, duplicate person, cloned face, changing outfit, inconsistent clothing, "
            "watermark, text, signature, cropped, out of frame, "
            "low quality, worst quality, jpeg artifacts, "
            "cartoon, anime style, illustration, painting, drawing, sketch, "
            "anime, manga, comic, 2D art style, 3D render, CGI, "
            "plastic looking, toy-like, over-saturated, oversaturated colors"
        )

        # IP-Adapter FaceID disabled

        # Step 2b: Batch generate all frames
        # Register cross-attention hooks for bounded SCA mask extraction
        self._register_cross_attention_hooks()
        try:
            batch_kwargs = {
                "prompt": prompts,
                "negative_prompt": [negative_prompt] * num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": gen_params.get("num_steps", 35),
                "guidance_scale": gen_params.get("guidance_scale", 7.5),
                "generator": generator,
            }
            if ip_adapter_image_embeds:
                batch_kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds

            total_steps = gen_params.get("num_steps", 35)
            self._set_sca_step_state(0, total_steps)
            import inspect
            if "callback_on_step_end" in inspect.signature(self.base_pipe.__call__).parameters:
                def _step_callback(_pipe, step_index, _timestep, callback_kwargs):
                    self._set_sca_step_state(step_index + 1, total_steps)
                    # After layout is formed (~30%), build and set bounded attention masks
                    if step_index == int(total_steps * 0.3):
                        masks = self._build_character_masks_from_attention(
                            prompts, height, width, production_board=production_board
                        )
                        if masks is not None and self._attn_processor is not None:
                            if isinstance(self._attn_processor, dict):
                                for proc in self._attn_processor.values():
                                    if isinstance(proc, ConsistentSelfAttentionProcessor):
                                        proc.set_bounded_masks(masks)
                    return callback_kwargs
                batch_kwargs["callback_on_step_end"] = _step_callback

            print(f"\n[Generate] Running batch inference ({num_frames} frames)...")
            output = self.base_pipe(**batch_kwargs)
            print(f"[Generate] Batch inference complete!")

            # Step 2c: Process batch output
            for i, panel in enumerate(production_board.panels):
                current_image = output.images[i]

                # Remove white/gray borders (SDXL VAE artifact)
                try:
                    current_image = remove_white_borders(current_image, threshold=180)
                except Exception:
                    pass

                all_images.append(current_image)
                print(f"[Frame {i+1}/{num_frames}] Completed")

            # Update memory once with the full batch
            for img in all_images:
                self._update_memory(img)

        except Exception as e:
            print(f"[Generate] Batch generation failed: {e}")
            print("[Generate] Falling back to sequential generation...")
            import traceback
            traceback.print_exc()

            # Fallback: generate one by one
            for i, panel in enumerate(production_board.panels):
                prompt = prompts[i] if i < len(prompts) else ""
                try:
                    call_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "height": height,
                        "width": width,
                        "num_inference_steps": gen_params.get("num_steps", 35),
                        "guidance_scale": gen_params.get("guidance_scale", 7.5),
                        "generator": generator,
                    }
                    single_ip_adapter_embeds = self._slice_ip_adapter_embeds(ip_adapter_image_embeds, i, num_frames)
                    if single_ip_adapter_embeds:
                        call_kwargs["ip_adapter_image_embeds"] = single_ip_adapter_embeds
                    frame_output = self.base_pipe(**call_kwargs)
                    current_image = frame_output.images[0]
                    try:
                        current_image = remove_white_borders(current_image, threshold=180)
                    except Exception:
                        pass
                    all_images.append(current_image)
                    self._update_memory(current_image)
                    print(f"[Frame {i+1}/{num_frames}] Completed (sequential fallback)")
                except Exception as e2:
                    print(f"[Generate] Sequential frame {i+1} also failed: {e2}")
                    placeholder = Image.new('RGB', (height, width), color=(128, 128, 128))
                    all_images.append(placeholder)

        # Clean up cross-attention hooks after generation to prevent memory leaks
        self._cleanup_cross_attention_hooks()

        print(f"\n{'=' * 60}")
        print(f"[Generate] Story generation complete! Generated {len(all_images)} frames")
        print(f"{'=' * 60}\n")

        if return_portraits:
            return all_images, portraits
        return all_images, None

    def _update_memory(self, image: Image.Image):
        """
        Update memory bank with features from generated image
        FIXED: Proper device handling for model_cpu_offload scenarios

        Args:
            image: PIL Image to extract features from
        """
        # Skip memory update if consistency is disabled
        if self._attn_processor is None and self._memory_bank is None:
            return
            
        try:
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

            # Create tensor on CPU first, then move to the same device as VAE
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                # Get VAE device safely
                vae = self.base_pipe.vae
                
                # Move VAE to CPU for encoding, then back (safer for CPU offload)
                vae_device = next(vae.parameters()).device if hasattr(vae, 'parameters') and len(list(vae.parameters())) > 0 else torch.device('cpu')
                
                # Move tensor to VAE's device
                img_for_vae = img_tensor.to(device=vae_device, dtype=vae.dtype)
                
                # Encode
                latent = vae.encode(
                    img_for_vae * 2 - 1  # Normalize to [-1, 1]
                ).latent_dist.sample()
                latent = latent * vae.config.scaling_factor

            # Move latent to pipeline device for consistency processing
            latent = latent.to(dtype=self.dtype, device=self.device)

            # Update attention processor memory
            if self._attn_processor is not None:
                b, c, h, w = latent.shape
                features = latent.view(b, c, h * w).permute(0, 2, 1)
                if isinstance(self._attn_processor, dict):
                    for proc in self._attn_processor.values():
                        proc.update_memory(features)
                else:
                    self._attn_processor.update_memory(features)

            # Update memory bank
            if self._memory_bank is not None:
                self._memory_bank.update(latent)

        except Exception as e:
            # Silently skip memory update errors - don't disrupt generation
            pass

    def save_story_images(
        self,
        images: List[Image.Image],
        story_id: str,
        panels: List[Panel],
        output_dir: str = "outputs/test_results"
    ) -> List[str]:
        """
        Save generated story images

        Args:
            images: List of generated PIL Images
            story_id: Story identifier
            panels: List of panels for metadata
            output_dir: Output directory

        Returns:
            List of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / f"{story_id}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, (img, panel) in enumerate(zip(images, panels)):
            # Save individual frame
            filename = f"frame_{i+1:02d}_{panel.shot_type}.png"
            filepath = save_dir / filename
            img.save(filepath)
            saved_paths.append(str(filepath))
            print(f"[Save] Saved: {filepath.name}")

        # Create and save storyboard
        storyboard = self._create_storyboard(images, panels)
        storyboard_path = save_dir / "storyboard.png"
        storyboard.save(storyboard_path)
        print(f"[Save] Storyboard saved: {storyboard_path.name}")

        return saved_paths

    def _create_storyboard(
        self,
        images: List[Image.Image],
        panels: List[Panel]
    ) -> Image.Image:
        """Create horizontal storyboard from generated images"""
        from PIL import ImageDraw, ImageFont

        # Target size
        target_height = 768
        target_width = int(target_height * 0.8)

        # Resize all images
        resized = [
            img.resize((target_width, target_height), Image.LANCZOS)
            for img in images
        ]

        # Horizontal layout
        spacing = 20
        total_width = sum(img.width for img in resized) + spacing * (len(resized) - 1)
        storyboard_height = target_height + 80

        storyboard = Image.new('RGB', (total_width, storyboard_height), color=(40, 40, 40))
        draw = ImageDraw.Draw(storyboard)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        x_offset = 0
        for i, (img, panel) in enumerate(zip(resized, panels)):
            storyboard.paste(img, (x_offset, 0))

            # Add frame number and scene description
            text = f"Scene {panel.panel_id}: {panel.shot_type}"
            draw.text((x_offset + 10, target_height + 10), text, fill=(200, 200, 200), font=font)

            x_offset += img.width + spacing

        return storyboard
