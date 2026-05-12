"""
Shared generation defaults for StoryGen entrypoints.

The goal is to keep SCA / face-lock / prompt-contract defaults aligned across
batch, smoke, and interactive runners so evaluation deltas are comparable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable


DEFAULT_GOLDEN_STORY_IDS = (
    "03",
    "11",
    "15",
    "16",
    "17",
    "18",
    "extra_06",
    "extra_07",
    "extra_11",
    "extra_12",
)


DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "device": "cuda",
    "use_fp16": True,
    "consistency_mode": "storydiffusion",
    "consistency_strength": 0.20,
    "sca_window_size": 1,
    "sca_start_ratio": 0.10,
    "enable_shot_type_sca_gating": True,
    "sca_min_pair_weight": 0.02,
    "sca_count_mismatch_penalty": 0.26,
    "sca_multi_character_penalty": 0.25,
    "sca_wide_shot_penalty": 0.18,
    "sca_scene_change_penalty": 0.08,
    "enable_face_lock": False,
    "face_lock_single_scale": 0.24,
    "face_lock_multi_scale": 0.0,
    "enable_entity_prompt_contract": True,
    "enable_role_bound_multi_character_anchors": True,
    "enable_vehicle_interior_prompting": True,
    "enable_terminal_dish_prompting": True,
    "memory_bank_size": 4,
    "memory_bank_capacity": 5,
    "memory_decay_factor": 0.9,
    "generation_params": {
        "num_steps": 40,
        "guidance_scale": 7.5,
    },
    "height": 1024,
    "width": 1024,
    "enable_model_cpu_offload": True,
}


def _merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_generation_config(
    overrides: Dict[str, Any] | None = None,
    *,
    device: str | None = None,
) -> Dict[str, Any]:
    config = _merge_dict(DEFAULT_GENERATION_CONFIG, overrides or {})
    if device is not None:
        config["device"] = device
    return config


def get_golden_story_ids(extra_ids: Iterable[str] | None = None) -> list[str]:
    story_ids = list(DEFAULT_GOLDEN_STORY_IDS)
    for story_id in extra_ids or []:
        if story_id not in story_ids:
            story_ids.append(story_id)
    return story_ids
