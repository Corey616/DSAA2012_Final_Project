---
name: full-eval
description: >-
  Run the full StoryGen evaluation loop after parser generator or evaluator
  changes. Use this when you need quantitative batch metrics a versioned
  qualitative diagnostic and a proceed-or-block verdict.
license: MIT
compatibility:
  - python >= 3.10
  - torch >= 2.0
metadata:
  priority: high
  last_updated: 2026-05-11
---

Use this workflow for changes that affect story state generation image generation or evaluation logic.

1. Run the full quantitative batch:
   - python3 scripts/parallel_batch_runner.py --max-gpus 4
2. Read aggregate results from:
   - outputs/taskA_batch/batch_summary.json
3. Run the qualitative diagnostic via the vision-eval skill.
4. Compare the result against the current baseline recorded in .opencode/project_state.md.
5. If the change regresses overall quality or introduces new critical failure tags block the change and hand the results to experiment-reviewer.
6. After the verdict use state-update to persist the new state.
