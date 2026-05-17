---
name: sca-impl
description: >-
  Implement or modify StoryGen consistent self-attention and batched generation.
  Use this when working on cross-frame consistency logic in pipeline.py or
  consistent_self_attn.py.
license: MIT
compatibility:
  - python >= 3.10
  - torch >= 2.0
metadata:
  priority: high
  last_updated: 2026-05-08
---

Use this skill when changing StoryGen SCA behavior.

1. Read:
   - storygen/core_generator/attention/consistent_self_attn.py
   - storygen/core_generator/pipeline.py
   - storygen/core_generator/memory_bank.py
2. Preserve the core invariants unless the task explicitly redefines them:
   - Q stays unchanged
   - K/V provide cross-frame sharing
   - batch generation is required for SCA
   - no manual 4D attention reshaping
3. Keep consistency scheduling layer weighting and window behavior explicit and testable.
4. Validate on a smoke story first then run full-eval.
