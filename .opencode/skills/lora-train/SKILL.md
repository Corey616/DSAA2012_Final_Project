---
name: lora-train
description: >-
  Plan or implement the non-human character path for StoryGen. Use this when
  robot animal or creature cases need model routing LoRA strategy or
  prompt/schema support.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: medium
  last_updated: 2026-05-08
---

Use this skill when non-human character generation is the problem surface.

1. Inspect representative non-human stories first:
   - storygen/data/TaskA/13.txt
   - storygen/data/TaskA/03.txt
   - storygen/data/TaskA/extra_06.txt
   - storygen/data/TaskA/extra_10.txt
2. Read the current pipeline and parser path before changing model routing.
3. Keep the distinction clear between prompt/schema fixes model selection changes and LoRA training recommendations.
4. Do not claim the issue is solved until the affected non-human cases are re-evaluated.
