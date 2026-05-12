#!/usr/bin/env python3
"""
Compare a candidate batch against a trusted batch.
Convention: pair stories by the stem of evaluation.json["script"] (fallback: subdirectory name).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storygen.evaluation_hub.failure_atlas import load_story_eval_dir


def _load_run(run_dir: Path) -> tuple[dict[str, dict], list[dict[str, str]]]:
    stories: dict[str, dict] = {}
    skipped: list[dict[str, str]] = []
    for story_dir in sorted(run_dir.iterdir()):
        if not story_dir.is_dir():
            continue
        try:
            story = load_story_eval_dir(story_dir)
        except FileNotFoundError as exc:
            skipped.append({"story_id": story_dir.name, "reason": str(exc)})
            continue
        stories[story["story_id"]] = story
    return stories, skipped


def _story_delta(candidate: dict, trusted: dict) -> dict:
    candidate_tags = set(candidate.get("failure_tags", []))
    trusted_tags = set(trusted.get("failure_tags", []))
    candidate_missing = set(candidate.get("missing_categories", []))
    trusted_missing = set(trusted.get("missing_categories", []))

    return {
        "story_id": candidate["story_id"],
        "delta_overall_score": round(candidate["overall_score"] - trusted["overall_score"], 4),
        "delta_project_score": round(
            candidate["project_rubric"]["overall"] - trusted["project_rubric"]["overall"],
            4,
        ),
        "delta_alignment_score": round(
            candidate["state_alignment_score"] - trusted["state_alignment_score"],
            4,
        ),
        "candidate_overall_score": candidate["overall_score"],
        "trusted_overall_score": trusted["overall_score"],
        "candidate_project_score": candidate["project_rubric"]["overall"],
        "trusted_project_score": trusted["project_rubric"]["overall"],
        "candidate_alignment_score": candidate["state_alignment_score"],
        "trusted_alignment_score": trusted["state_alignment_score"],
        "candidate_parser_variant": candidate.get("parser_variant"),
        "trusted_parser_variant": trusted.get("parser_variant"),
        "added_tags": sorted(candidate_tags - trusted_tags),
        "removed_tags": sorted(trusted_tags - candidate_tags),
        "unchanged_tags": sorted(candidate_tags & trusted_tags),
        "added_missing_categories": sorted(candidate_missing - trusted_missing),
        "resolved_missing_categories": sorted(trusted_missing - candidate_missing),
    }


def build_comparison(candidate_dir: Path, trusted_dir: Path) -> dict:
    candidate_stories, candidate_skipped = _load_run(candidate_dir)
    trusted_stories, trusted_skipped = _load_run(trusted_dir)

    candidate_ids = set(candidate_stories)
    trusted_ids = set(trusted_stories)
    shared_ids = sorted(candidate_ids & trusted_ids)

    story_deltas = [
        _story_delta(candidate_stories[story_id], trusted_stories[story_id])
        for story_id in shared_ids
    ]
    story_deltas.sort(key=lambda item: item["delta_overall_score"])

    regressions = [item for item in story_deltas if item["delta_overall_score"] < 0]
    improvements = sorted(
        [item for item in story_deltas if item["delta_overall_score"] > 0],
        key=lambda item: item["delta_overall_score"],
        reverse=True,
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "candidate_dir": str(candidate_dir),
        "trusted_dir": str(trusted_dir),
        "shared_story_count": len(shared_ids),
        "only_in_candidate": sorted(candidate_ids - trusted_ids),
        "only_in_trusted": sorted(trusted_ids - candidate_ids),
        "candidate_skipped": candidate_skipped,
        "trusted_skipped": trusted_skipped,
        "summary": {
            "improved_story_count": len(improvements),
            "regressed_story_count": len(regressions),
            "unchanged_story_count": sum(1 for item in story_deltas if item["delta_overall_score"] == 0),
        },
        "top_improvements": improvements[:10],
        "top_regressions": regressions[:10],
        "stories": story_deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare candidate and trusted StoryGen batch outputs")
    parser.add_argument("--candidate", required=True, help="Candidate batch output directory")
    parser.add_argument("--trusted", required=True, help="Trusted batch output directory")
    parser.add_argument("--output", help="Where to write comparison.json (default: <candidate>/comparison.json)")
    args = parser.parse_args()

    candidate_dir = Path(args.candidate)
    trusted_dir = Path(args.trusted)
    output_path = Path(args.output) if args.output else candidate_dir / "comparison.json"

    comparison = build_comparison(candidate_dir, trusted_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    summary = comparison["summary"]
    print(
        "Compared "
        f"{comparison['shared_story_count']} shared stories: "
        f"{summary['improved_story_count']} improved, "
        f"{summary['regressed_story_count']} regressed, "
        f"{summary['unchanged_story_count']} unchanged. "
        f"Saved to {output_path}"
    )
    if comparison["only_in_candidate"] or comparison["only_in_trusted"]:
        print(
            "Unmatched stories - "
            f"candidate only: {comparison['only_in_candidate']}, "
            f"trusted only: {comparison['only_in_trusted']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
