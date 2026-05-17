---
name: state-diagnosis
description: >-
  Diagnose why a StoryGen case failed to satisfy story state. Use this when
  outputs look wrong and you need to map the visual failure back to story_state
  render_plan prompt compilation or consistency control.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: high
  last_updated: 2026-05-08
---

Use this skill for targeted failure analysis.

1. Read for the same story ID:
   - outputs/taskA_batch/story_id/story_state.json
   - outputs/taskA_batch/story_id/render_plan.json
   - outputs/taskA_batch/story_id/production_board.json
   - outputs/taskA_batch/story_id/evaluation.json
2. Identify whether the failure starts in parsing prompt compilation generation control or evaluation.
3. Classify the result with concrete tags:
   - face_drift | wardrobe_drift | count_drift | layout_failure | scene_overwrite | relational_prop_loss | artifact_failure | prompt_overspecification
4. End with the smallest next change that can be tested on a smoke set before running a full batch.
