"""
Project-aligned offline rubric scoring.
This augments the existing CLIP/LPIPS gate without replacing it.
"""

from __future__ import annotations

from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalized_clip(clip_score: float) -> float:
    # Typical StoryGen CLIP scores cluster around 0.25-0.35; rescale without changing the raw metric.
    return _clamp(clip_score / 0.4)


def _entity_continuity_score(story_eval: dict[str, Any]) -> float:
    entity_summary = story_eval.get("entity_summary", {}) or {}
    failure_tags = set(story_eval.get("failure_tags", []) or [])

    penalty = 0.0
    penalty += 0.18 * min(int(entity_summary.get("count_misses", 0) or 0), 2)
    penalty += 0.08 * min(int(entity_summary.get("wardrobe_misses", 0) or 0), 3)
    penalty += 0.08 * min(int(entity_summary.get("identity_detail_misses", 0) or 0), 3)
    penalty += 0.18 if "count_drift" in failure_tags else 0.0
    penalty += 0.12 if "identity_drift" in failure_tags else 0.0
    penalty += 0.08 if "face_drift" in failure_tags else 0.0
    penalty += 0.08 if "wardrobe_drift" in failure_tags else 0.0
    return _clamp(1.0 - penalty)


def _story_continuity_score(story_eval: dict[str, Any]) -> float:
    failure_tags = set(story_eval.get("failure_tags", []) or [])

    penalty = 0.0
    penalty += 0.16 if "prompt_mismatch" in failure_tags else 0.0
    penalty += 0.12 if "layout_failure" in failure_tags else 0.0
    penalty += 0.08 if "scene_drift" in failure_tags else 0.0
    penalty += 0.08 if "prop_drop" in failure_tags else 0.0
    penalty += 0.06 if "artifact_failure" in failure_tags else 0.0
    return _clamp(1.0 - penalty)


def _aesthetic_reliability_score(story_eval: dict[str, Any]) -> float:
    clip_component = _normalized_clip(float(story_eval.get("avg_clip_score", 0.0) or 0.0))
    consistency_component = _clamp(float(story_eval.get("avg_consistency", 0.0) or 0.0))
    return _clamp((0.6 * clip_component) + (0.4 * consistency_component))


def compute_project_rubric(story_eval: dict[str, Any]) -> dict[str, float]:
    panel_fidelity = _clamp(float(story_eval.get("state_alignment_score", 0.0) or 0.0))
    entity_continuity = _entity_continuity_score(story_eval)
    story_continuity = _story_continuity_score(story_eval)
    aesthetic_reliability = _aesthetic_reliability_score(story_eval)

    overall = (
        0.35 * panel_fidelity
        + 0.30 * entity_continuity
        + 0.20 * story_continuity
        + 0.15 * aesthetic_reliability
    )
    return {
        "panel_fidelity": round(panel_fidelity, 4),
        "entity_continuity": round(entity_continuity, 4),
        "story_continuity": round(story_continuity, 4),
        "aesthetic_reliability": round(aesthetic_reliability, 4),
        "overall": round(overall, 4),
    }

