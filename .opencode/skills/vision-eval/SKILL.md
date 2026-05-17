---
name: vision-eval
description: >-
  Run the StoryGen qualitative diagnostic over generated stories. Use this after
  quantitative evaluation to inspect character consistency scene coherence
  prompt alignment temporal flow style uniformity and visible artifacts.
license: MIT
compatibility:
  - python >= 3.10
  - torch >= 2.0
metadata:
  priority: high
  last_updated: 2026-05-08
---

Use this skill after a quantitative run or when investigating a visual regression.

1. Prefer the local storygen.evaluation_hub.vision_eval entry points.
2. For a single story use:
   - python3 -m storygen.evaluation_hub.vision_eval 11
3. For a full batch use a version-tagged run:
   - python3 -c "from storygen.evaluation_hub.vision_eval import evaluate_all_stories; print(evaluate_all_stories(base_dir='outputs/taskA_batch', version_tag='candidate'))"
4. Read or update version comparisons with VersionTracker when diagnosing regressions.
5. Report concrete failure surfaces not generic statements: face drift count drift scene overwrite layout failure relational prop loss or artifact quality issues.
