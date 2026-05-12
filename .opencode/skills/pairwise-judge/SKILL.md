---
name: pairwise-judge
description: >-
  Compare a candidate StoryGen batch against the trusted batch. Use this when a
  new run needs story-by-story and dimension-by-dimension deltas not just a new
  average score.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: medium
  last_updated: 2026-05-08
---

Use this skill after a candidate batch is available.

1. Run the offline comparator:
   - python3 scripts/compare_runs.py --candidate candidate_dir --trusted trusted_dir
2. Read candidate_dir/comparison.json or the explicit --output path
3. Focus on:
   - top_improvements
   - top_regressions
   - delta_project_score
   - delta_overall_score
   - added_tags
   - removed_tags
   - added_missing_categories
   - resolved_missing_categories
4. Treat unmatched stories as a blocking setup issue not a model win:
   - only_in_candidate
   - only_in_trusted
5. End with one verdict: proceed | fix-and-retry | rollback
6. If the candidate did not improve its intended target stories do not promote it even if the global average increased.
