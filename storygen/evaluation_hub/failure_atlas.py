"""
Offline evaluation aggregation utilities.
Builds a lightweight failure atlas from saved story outputs without invoking GPU models.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from storygen.evaluation_hub.project_rubric import compute_project_rubric


CATEGORY_TO_TAG = {
    "characters": "identity_drift",
    "count": "count_drift",
    "wardrobe": "wardrobe_drift",
    "identity_detail": "face_drift",
    "setting": "scene_drift",
    "continuity": "prop_drop",
    "layout": "layout_failure",
    "action": "prompt_mismatch",
}

ENTITY_SUMMARY_KEYS = {
    "count": "count_misses",
    "wardrobe": "wardrobe_misses",
    "identity_detail": "identity_detail_misses",
    "layout": "layout_misses",
}

PROMPT_RISK_TAGS = {
    "multi_human_gender_binding_risk",
    "multi_human_wardrobe_binding_risk",
    "vehicle_interior_risk",
    "terminal_dish_binding_risk",
}


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _collect_story_prompt_risks(story_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    story_state = _load_json(story_dir / "story_state.json")
    render_plan = _load_json(story_dir / "render_plan.json")
    if not isinstance(story_state, dict) or not isinstance(render_plan, list):
        return [], {}

    panel_state_map = {
        int(panel_state.get("panel_id")): panel_state
        for panel_state in story_state.get("panel_states", [])
        if isinstance(panel_state, dict) and panel_state.get("panel_id") is not None
    }
    prompt_risks: list[dict[str, Any]] = []
    compiler_annotation_counts: dict[str, int] = {}

    def add_risk(panel_id: int, tag: str, details: str) -> None:
        if tag not in PROMPT_RISK_TAGS:
            return
        prompt_risks.append({
            "panel_id": panel_id,
            "tag": tag,
            "details": details,
        })

    for panel_plan in render_plan:
        if not isinstance(panel_plan, dict):
            continue
        panel_id = int(panel_plan.get("panel_id", 0) or 0)
        prompt_slots = panel_plan.get("prompt_slots", []) or []
        compiled_prompt = str(panel_plan.get("compiled_prompt", "") or "")
        compiled_lower = compiled_prompt.lower()
        panel_state = panel_state_map.get(panel_id, {})
        action_beats = " ".join(str(item) for item in panel_state.get("action_beats", [])).lower()
        panel_entities = panel_state.get("panel_entities", []) or []

        for annotation in panel_plan.get("compiler_annotations", []) or []:
            label = str(annotation).strip()
            if label:
                compiler_annotation_counts[label] = compiler_annotation_counts.get(label, 0) + 1

        human_entities = [
            entity for entity in panel_entities
            if str(entity.get("entity_type", "")).lower() == "human"
        ]
        identity_slot_texts = [
            str(slot.get("text", "")).lower()
            for slot in prompt_slots
            if str(slot.get("name", "")) == "identity_anchor"
        ]
        multi_human_panel = len(human_entities) >= 2
        role_bound_identities = sum(
            1 for text in identity_slot_texts
            if "on the left" in text or "on the right" in text
        )
        if multi_human_panel and role_bound_identities < len(human_entities):
            if any(str(entity.get("gender", "")).lower() in {"male", "female"} for entity in human_entities):
                add_risk(
                    panel_id,
                    "multi_human_gender_binding_risk",
                    "multi-human panel keeps gender cues, but identity anchors are not role-bound to left/right positions.",
                )
            if any(entity.get("wardrobe_terms") for entity in human_entities):
                add_risk(
                    panel_id,
                    "multi_human_wardrobe_binding_risk",
                    "multi-human panel keeps wardrobe cues, but identity anchors are not role-bound to left/right positions.",
                )

        vehicle_context = " ".join([
            compiled_lower,
            " ".join(str(item).lower() for item in panel_state.get("must_show", [])),
            action_beats,
        ])
        if "car" in vehicle_context and any(token in action_beats for token in ("drive", "drives", "driving", "scenery outside", "looks at the scenery outside")):
            interior_tokens = (
                "inside the car",
                "driver's seat",
                "steering wheel",
                "car window",
                "windshield",
            )
            if not any(token in compiled_lower for token in interior_tokens):
                add_risk(
                    panel_id,
                    "vehicle_interior_risk",
                    "car-driving panel lacks explicit inside-car / driver-seat / window binding in the compiled prompt.",
                )

        prop_anchor_text = " ".join(
            str(slot.get("text", "")).lower()
            for slot in prompt_slots
            if str(slot.get("name", "")) == "prop_anchor"
        )
        if "serve" in action_beats or "finished dish" in action_beats:
            terminal_food_tokens = ("prepared food", "finished dish", "plated", "dish", "bowl", "plate")
            prep_tokens = ("vegetables", "cutting board", "chef's knife", "knife")
            if not any(token in prop_anchor_text for token in terminal_food_tokens) or any(token in prop_anchor_text for token in prep_tokens):
                add_risk(
                    panel_id,
                    "terminal_dish_binding_risk",
                    "serve-finished-dish panel keeps prep-stage props or lacks plated/prepared-dish prop cues.",
                )

    return prompt_risks, compiler_annotation_counts


def _story_id_from_eval(story_dir: Path, eval_data: dict[str, Any]) -> str:
    script = str(eval_data.get("script", "") or "")
    if script:
        return Path(script).stem
    return story_dir.name


def _build_panel_summary(panel_alignment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for panel in panel_alignment:
        missing_by_category = panel.get("missing_by_category", {}) or {}
        summary.append({
            "panel_id": panel.get("panel_id"),
            "score": panel.get("score", 0.0),
            "missing_categories": sorted(missing_by_category.keys()),
            "missing_terms": [
                f"{category}: {term}"
                for category, terms in missing_by_category.items()
                for term in terms
            ][:6],
        })
    return summary


def _derive_failure_tags_from_panel_alignment(panel_alignment: list[dict[str, Any]]) -> list[str]:
    tags = set()
    for panel in panel_alignment:
        missing_by_category = panel.get("missing_by_category", {}) or {}
        for category in missing_by_category:
            tag = CATEGORY_TO_TAG.get(category)
            if tag:
                tags.add(tag)
    return sorted(tags)


def _derive_entity_summary_from_panel_alignment(panel_alignment: list[dict[str, Any]]) -> dict[str, int]:
    entity_summary = {
        "count_misses": 0,
        "wardrobe_misses": 0,
        "identity_detail_misses": 0,
        "layout_misses": 0,
    }
    for panel in panel_alignment:
        missing_by_category = panel.get("missing_by_category", {}) or {}
        for category, terms in missing_by_category.items():
            key = ENTITY_SUMMARY_KEYS.get(category)
            if key:
                entity_summary[key] += len(terms)
    return entity_summary


def load_story_eval_dir(story_dir: Path) -> dict[str, Any]:
    eval_data = _load_json(story_dir / "evaluation.json")
    if not isinstance(eval_data, dict):
        raise FileNotFoundError(f"missing evaluation.json in {story_dir}")
    prompt_risks, compiler_annotation_counts = _collect_story_prompt_risks(story_dir)

    panel_alignment = _load_json(story_dir / "panel_alignment.json")
    if not isinstance(panel_alignment, list):
        panel_alignment = eval_data.get("panel_alignment", [])
    if not isinstance(panel_alignment, list):
        panel_alignment = []

    panel_alignment_summary = eval_data.get("panel_alignment_summary", [])
    if not isinstance(panel_alignment_summary, list) or not panel_alignment_summary:
        panel_alignment_summary = _build_panel_summary(panel_alignment)

    failure_tags = eval_data.get("failure_tags", [])
    if not isinstance(failure_tags, list) or not failure_tags:
        failure_tags = _derive_failure_tags_from_panel_alignment(panel_alignment)

    entity_summary = eval_data.get("entity_summary", {})
    if not isinstance(entity_summary, dict) or not entity_summary:
        entity_summary = _derive_entity_summary_from_panel_alignment(panel_alignment)

    vision_judge = eval_data.get("vision_judge", {})
    if not isinstance(vision_judge, dict):
        vision_judge = {}

    missing_categories = sorted({
        category
        for panel in panel_alignment_summary
        for category in (panel.get("missing_categories", []) or [])
    })

    story_eval = {
        "story_id": _story_id_from_eval(story_dir, eval_data),
        "story_dir": str(story_dir),
        "script": eval_data.get("script", ""),
        "overall_score": float(eval_data.get("overall_score", 0.0) or 0.0),
        "avg_clip_score": float(eval_data.get("avg_clip_score", 0.0) or 0.0),
        "avg_consistency": float(eval_data.get("avg_consistency", 0.0) or 0.0),
        "state_alignment_score": float(eval_data.get("state_alignment_score", 0.0) or 0.0),
        "failure_tags": sorted({str(tag) for tag in failure_tags if str(tag).strip()}),
        "entity_summary": entity_summary,
        "panel_alignment": panel_alignment,
        "panel_alignment_summary": panel_alignment_summary,
        "missing_categories": missing_categories,
        "parser_variant": eval_data.get("parser_variant"),
        "prompt_risks": prompt_risks,
        "compiler_annotation_counts": compiler_annotation_counts,
        "vision_judge": vision_judge,
    }
    story_eval["project_rubric"] = compute_project_rubric(story_eval)
    return story_eval


def build_failure_atlas(story_dirs: Iterable[Path]) -> dict[str, Any]:
    tag_counts: dict[str, int] = {}
    tag_examples: dict[str, list[dict[str, Any]]] = {}
    prompt_risk_counts: dict[str, int] = {}
    prompt_risk_examples: dict[str, list[dict[str, Any]]] = {}
    compiler_annotation_counts: dict[str, int] = {}
    vision_judge_tag_counts: dict[str, int] = {}
    vision_judge_examples: dict[str, list[dict[str, Any]]] = {}
    category_miss_counts: dict[str, int] = {}
    entity_miss_totals = {
        "count_misses": 0,
        "wardrobe_misses": 0,
        "identity_detail_misses": 0,
        "layout_misses": 0,
    }
    per_story: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    for story_dir in sorted(story_dirs):
        if not story_dir.is_dir():
            continue
        try:
            story_data = load_story_eval_dir(story_dir)
        except FileNotFoundError as exc:
            skipped.append({"story_id": story_dir.name, "reason": str(exc)})
            continue

        story_id = story_data["story_id"]
        per_story[story_id] = {
            "overall_score": story_data["overall_score"],
            "avg_clip_score": story_data["avg_clip_score"],
            "avg_consistency": story_data["avg_consistency"],
            "state_alignment_score": story_data["state_alignment_score"],
            "project_rubric": story_data["project_rubric"],
            "failure_tags": story_data["failure_tags"],
            "missing_categories": story_data["missing_categories"],
            "parser_variant": story_data["parser_variant"],
            "prompt_risks": story_data["prompt_risks"],
            "vision_judge_tags": story_data["vision_judge"].get("issue_tags", []),
            "vision_judge_confidence": story_data["vision_judge"].get("confidence"),
        }

        for annotation, count in story_data["compiler_annotation_counts"].items():
            compiler_annotation_counts[annotation] = compiler_annotation_counts.get(annotation, 0) + int(count or 0)

        for key, value in story_data["entity_summary"].items():
            entity_miss_totals[key] = entity_miss_totals.get(key, 0) + int(value or 0)

        for tag in story_data["failure_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        for panel in story_data["panel_alignment"]:
            missing_by_category = panel.get("missing_by_category", {}) or {}
            for category, terms in missing_by_category.items():
                category_miss_counts[category] = category_miss_counts.get(category, 0) + len(terms)
                tag = CATEGORY_TO_TAG.get(category)
                if not tag:
                    continue
                examples = tag_examples.setdefault(tag, [])
                if len(examples) >= 5:
                    continue
                examples.append({
                    "story_id": story_id,
                    "panel_id": panel.get("panel_id"),
                    "missing_terms": list(terms)[:3],
                })

        for risk in story_data["prompt_risks"]:
            tag = str(risk.get("tag", "")).strip()
            if not tag:
                continue
            prompt_risk_counts[tag] = prompt_risk_counts.get(tag, 0) + 1
            examples = prompt_risk_examples.setdefault(tag, [])
            if len(examples) >= 5:
                continue
            examples.append({
                "story_id": story_id,
                "panel_id": risk.get("panel_id"),
                "details": risk.get("details", ""),
            })

        vision_judge = story_data["vision_judge"]
        if vision_judge.get("confidence") != "low":
            for tag in vision_judge.get("issue_tags", []) or []:
                stable_tag = str(tag).strip()
                if not stable_tag:
                    continue
                vision_judge_tag_counts[stable_tag] = vision_judge_tag_counts.get(stable_tag, 0) + 1
            for panel in vision_judge.get("panel_judgments", []) or []:
                for tag in panel.get("issue_tags", []) or []:
                    examples = vision_judge_examples.setdefault(str(tag), [])
                    if len(examples) >= 5:
                        continue
                    examples.append({
                        "story_id": story_id,
                        "panel_id": panel.get("panel_id"),
                        "source": "panel",
                    })
            for continuity in vision_judge.get("cross_panel_judgments", []) or []:
                for tag in continuity.get("issue_tags", []) or []:
                    examples = vision_judge_examples.setdefault(str(tag), [])
                    if len(examples) >= 5:
                        continue
                    examples.append({
                        "story_id": story_id,
                        "from_panel": continuity.get("from_panel"),
                        "to_panel": continuity.get("to_panel"),
                        "source": "cross_panel",
                    })

    lowest_stories = [
        {
            "story_id": story_id,
            "overall_score": story_data["overall_score"],
            "project_score": story_data["project_rubric"]["overall"],
            "failure_tags": story_data["failure_tags"],
            "missing_categories": story_data["missing_categories"],
        }
        for story_id, story_data in sorted(
            per_story.items(),
            key=lambda item: item[1]["overall_score"],
        )[:10]
    ]

    lowest_project_stories = [
        {
            "story_id": story_id,
            "project_score": story_data["project_rubric"]["overall"],
            "overall_score": story_data["overall_score"],
            "failure_tags": story_data["failure_tags"],
            "missing_categories": story_data["missing_categories"],
        }
        for story_id, story_data in sorted(
            per_story.items(),
            key=lambda item: item[1]["project_rubric"]["overall"],
        )[:10]
    ]

    return {
        "generated_at": datetime.now().isoformat(),
        "story_count": len(per_story),
        "tag_counts": dict(sorted(tag_counts.items())),
        "tag_examples": tag_examples,
        "prompt_risk_counts": dict(sorted(prompt_risk_counts.items())),
        "prompt_risk_examples": prompt_risk_examples,
        "compiler_annotation_counts": dict(sorted(compiler_annotation_counts.items())),
        "vision_judge_tag_counts": dict(sorted(vision_judge_tag_counts.items())),
        "vision_judge_examples": vision_judge_examples,
        "category_miss_counts": dict(sorted(category_miss_counts.items())),
        "entity_miss_totals": entity_miss_totals,
        "lowest_stories": lowest_stories,
        "lowest_project_stories": lowest_project_stories,
        "per_story": per_story,
        "skipped": skipped,
    }
