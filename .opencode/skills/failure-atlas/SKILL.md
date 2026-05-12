---
name: failure-atlas
description: >-
  Summarize the weakest StoryGen stories and failure clusters from saved batch
  outputs. Use this after a batch run when you need actionable low-tail triage
  before changing code.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: high
  last_updated: 2026-05-08
---

Use this skill after full-eval or any batch run that wrote outputs/taskA_batch/.

1. Prefer the saved batch artifact first:
   - outputs/taskA_batch/failure_atlas.json
2. If it does not exist yet regenerate it offline:
   - python3 -c "from storygen.evaluation_hub.failure_atlas import build_failure_atlas; from pathlib import Path; import json; atlas = build_failure_atlas(Path('outputs/taskA_batch').iterdir()); Path('outputs/taskA_batch/failure_atlas.json').write_text(json.dumps(atlas, indent=2))"
3. Report concrete outputs not generic impressions:
   - lowest_project_stories
   - lowest_stories
   - tag_counts
   - category_miss_counts
   - tag_examples
4. Separate raw metric low tail from project-rubric low tail.
5. End with the smallest next tranche to test tied to a specific failure family:
   - count_drift | wardrobe_drift | face_drift | prompt_mismatch | layout_failure | non-human under-grounding
