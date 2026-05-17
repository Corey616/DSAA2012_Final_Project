---
name: face-lock
description: >-
  Implement identity conditioning for StoryGen without freezing scenes. Use this
  when improving facial or clothing consistency through portrait-conditioned
  features or IP-Adapter-style integration.
license: MIT
compatibility:
  - python >= 3.10
  - torch >= 2.0
metadata:
  priority: medium
  last_updated: 2026-05-08
---

Use this skill when working on identity locking.

1. Read:
   - `storygen/asset_anchor/character_portrait.py`
   - `storygen/core_generator/pipeline.py`
2. Reuse portrait-derived identity features where possible.
3. Do not use full-image IP-Adapter conditioning because it freezes scene structure.
4. Keep identity conditioning separate from scene/layout control so regressions stay diagnosable.
5. Validate on a focused smoke set before full-eval.
