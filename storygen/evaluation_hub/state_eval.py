"""
State-grounded evaluation utilities.
Compares compiled prompts and vision findings against StoryState artifacts.
"""

import re
from typing import Any, Dict, List


STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "in", "on", "at", "to", "of", "for",
    "from", "by", "near", "same", "still", "remain", "transition", "carry", "over",
    "his", "her", "their", "its", "he", "she", "they", "it", "under", "through",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).strip()


def _normalize_token(token: str) -> str:
    cleaned = _normalize_text(token)
    if cleaned.endswith("ing") and len(cleaned) > 5:
        cleaned = cleaned[:-3]
    elif cleaned.endswith("ed") and len(cleaned) > 4:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("es") and len(cleaned) > 4:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("s") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return cleaned


def _extract_keywords(term: str) -> List[str]:
    words = [_normalize_token(word) for word in _normalize_text(term).split()]
    keywords = [word for word in words if len(word) >= 3 and word not in STOPWORDS]
    return list(dict.fromkeys(keywords))


def _term_matches_prompt(term: str, prompt_text: str, category: str) -> bool:
    normalized_term = _normalize_text(term)
    normalized_prompt = _normalize_text(prompt_text)
    if normalized_term and normalized_term in normalized_prompt:
        return True

    keywords = _extract_keywords(term)
    if not keywords:
        return False

    prompt_keywords = set(_extract_keywords(prompt_text))
    overlap = [word for word in keywords if word in prompt_keywords]

    if category == "characters":
        return bool(overlap)
    if category in {"wardrobe", "identity_detail", "layout", "action"}:
        return bool(overlap)

    if len(keywords) == 1:
        return keywords[0] in prompt_keywords

    min_overlap = 2 if len(keywords) >= 3 else 1
    return len(overlap) >= min_overlap


def _count_matches_prompt(expected_count: int, prompt_text: str, visible_names: List[str]) -> bool:
    if expected_count <= 0:
        return True
    normalized_prompt = _normalize_text(prompt_text)
    count_words = {
        1: {"one", "1"},
        2: {"two", "2", "both"},
        3: {"three", "3"},
        4: {"four", "4"},
    }.get(expected_count, {str(expected_count)})
    if any(word in normalized_prompt.split() for word in count_words):
        return True
    matched_names = [
        name for name in visible_names
        if _normalize_text(name) and _normalize_text(name) in normalized_prompt
    ]
    return len(matched_names) >= expected_count


def build_state_eval_questions(story_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return machine-readable QA items derived from StoryState."""
    if not story_state:
        return []
    questions = story_state.get("story_questions_for_eval", [])
    return questions if isinstance(questions, list) else []


def _categorize_must_show_item(term: str) -> str:
    lowered = str(term or "").lower()
    if any(token in lowered for token in ("remain in", "transition", "carry over", "same ")):
        return "continuity"
    if any(token in lowered for token in ("left", "right", "center", "behind", "front", "next to", "beside")):
        return "layout"
    return "setting"


def _collect_panel_expected_items(panel_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    expected_items: List[Dict[str, Any]] = []
    characters_present = {str(item).strip().lower() for item in panel_state.get("characters_present", []) if str(item).strip()}
    action_beats = {str(item).strip().lower() for item in panel_state.get("action_beats", []) if str(item).strip()}
    for item in panel_state.get("characters_present", []):
        expected_items.append({"category": "characters", "term": item})
    for item in panel_state.get("must_show", [])[:4]:
        lowered = str(item).strip().lower()
        if lowered in characters_present or lowered in action_beats:
            continue
        expected_items.append({"category": _categorize_must_show_item(item), "term": item})

    visible_names = panel_state.get("characters_present", [])
    expected_count = int(panel_state.get("expected_count") or len(visible_names))
    if expected_count:
        expected_items.append({"category": "count", "term": str(expected_count)})

    for entity in panel_state.get("panel_entities", [])[:3]:
        for term in (entity.get("wardrobe_terms") or [])[:2]:
            expected_items.append({"category": "wardrobe", "term": term})
        for term in (entity.get("face_terms") or [])[:1]:
            expected_items.append({"category": "identity_detail", "term": term})
        if entity.get("position"):
            expected_items.append({"category": "layout", "term": f"{entity.get('name')} {entity.get('position')}"})
        if entity.get("action"):
            expected_items.append({"category": "action", "term": entity.get("action")})
    return expected_items


def build_panel_alignment_report(
    story_state: Dict[str, Any],
    prompts: List[str],
) -> List[Dict[str, Any]]:
    """Return one prompt-alignment report per panel with matched and missing obligations."""
    if not story_state:
        return []

    panel_states = story_state.get("panel_states", [])
    reports: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        prompt_text = (prompt or "").lower()
        panel_state = panel_states[index] if index < len(panel_states) else {}
        panel_id = panel_state.get("panel_id", index + 1)
        visible_names = panel_state.get("characters_present", [])
        expected_count = int(panel_state.get("expected_count") or len(visible_names))
        seen_terms = set()
        obligations = []
        matched = 0
        total = 0
        missing_by_category: Dict[str, List[str]] = {}

        for item in _collect_panel_expected_items(panel_state):
            term = str(item["term"]).strip()
            if len(term) < 3:
                continue
            dedupe_key = (item["category"], term.lower())
            if dedupe_key in seen_terms:
                continue
            seen_terms.add(dedupe_key)

            if item["category"] == "count":
                is_matched = _count_matches_prompt(expected_count, prompt_text, visible_names)
            else:
                is_matched = _term_matches_prompt(term, prompt_text, item["category"])
            total += 1
            if is_matched:
                matched += 1
            else:
                missing_by_category.setdefault(item["category"], []).append(term)
            obligations.append({
                "category": item["category"],
                "term": term,
                "matched": is_matched,
            })

        reports.append({
            "panel_id": panel_id,
            "prompt": prompt,
            "score": round(matched / total, 4) if total else 1.0,
            "matched": matched,
            "total": total,
            "obligations": obligations,
            "missing_by_category": missing_by_category,
            "missing_terms": [
                f"{category}: {term}"
                for category, terms in missing_by_category.items()
                for term in terms
            ],
        })

    return reports


def evaluate_prompt_state_alignment(
    story_state: Dict[str, Any],
    prompts: List[str],
) -> Dict[str, Any]:
    """
    Check whether compiled prompts preserve the key requirements encoded in StoryState.
    This is a fast structural check before vision-based inspection.
    """
    if not story_state:
        return {
            "score": 0.0,
            "checks": [],
            "issues": ["missing_story_state"],
            "panel_reports": [],
            "panel_summary": [],
            "entity_summary": {},
        }

    panel_reports = build_panel_alignment_report(story_state, prompts)
    checks = []
    passed = 0
    total = 0
    entity_summary = {
        "count_misses": 0,
        "wardrobe_misses": 0,
        "identity_detail_misses": 0,
        "layout_misses": 0,
    }

    for report in panel_reports:
        passed += report["matched"]
        total += report["total"]
        for obligation in report["obligations"]:
            checks.append({
                "panel_id": report["panel_id"],
                "category": obligation["category"],
                "term": obligation["term"],
                "matched": obligation["matched"],
            })
            if obligation["matched"]:
                continue
            if obligation["category"] == "count":
                entity_summary["count_misses"] += 1
            elif obligation["category"] == "wardrobe":
                entity_summary["wardrobe_misses"] += 1
            elif obligation["category"] == "identity_detail":
                entity_summary["identity_detail_misses"] += 1
            elif obligation["category"] == "layout":
                entity_summary["layout_misses"] += 1

    issues = []
    panel_summary = []
    for report in panel_reports:
        panel_summary.append({
            "panel_id": report["panel_id"],
            "score": report["score"],
            "missing_categories": sorted(report["missing_by_category"].keys()),
            "missing_terms": report["missing_terms"][:6],
        })
        for category, terms in report["missing_by_category"].items():
            for term in terms:
                issues.append(f"panel {report['panel_id']} missing {category}: {term}")

    score = round(passed / total, 4) if total else 1.0
    return {
        "score": score,
        "checks": checks,
        "issues": issues,
        "panel_reports": panel_reports,
        "panel_summary": panel_summary,
        "entity_summary": entity_summary,
    }


def derive_failure_tags(
    state_alignment: Dict[str, Any],
    vision_summary: str = "",
) -> List[str]:
    """Map alignment misses and vision findings to stable failure tags."""
    tags = set()

    for check in state_alignment.get("checks", []):
        if check.get("matched"):
            continue
        category = check.get("category")
        if category == "characters":
            tags.add("identity_drift")
        elif category == "count":
            tags.add("count_drift")
        elif category == "wardrobe":
            tags.add("wardrobe_drift")
        elif category == "identity_detail":
            tags.add("face_drift")
        elif category == "setting":
            tags.add("scene_drift")
        elif category == "continuity":
            tags.add("prop_drop")
        elif category == "layout":
            tags.add("layout_failure")
        elif category == "action":
            tags.add("prompt_mismatch")
        else:
            tags.add("prompt_overspecification")

    summary = (vision_summary or "").lower()
    if any(token in summary for token in ("artifact", "blur", "distort", "deform", "ghost")):
        tags.add("artifact_failure")
    if any(token in summary for token in ("scene", "background", "setting drift")):
        tags.add("scene_drift")
    if any(token in summary for token in ("character", "identity", "appearance", "clothing drift")):
        tags.add("identity_drift")

    return sorted(tags)
