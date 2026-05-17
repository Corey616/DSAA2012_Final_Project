---
name: storygen-code-map
description: >-
  Use this before modifying StoryGen parser generator attention or evaluation
  code. It tells the agent where the core files live and which invariants must
  be preserved.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: high
  last_updated: 2026-05-11
---

Use this skill to orient yourself in the StoryGen codebase before making non-trivial changes.

1. Start with the canonical code map in:
   - .opencode/skills/storygen-code/SKILL.md
2. Then inspect the files most relevant to the task:
   - storygen/script_director/llm_parser.py
   - storygen/core_generator/pipeline.py
   - storygen/core_generator/attention/consistent_self_attn.py
   - storygen/evaluation_hub/*.py
3. Preserve the current SCA invariants (Q unchanged, K/V expand, batch generation, 3D bmm/baddbmm, self-attention only, window attention, layer-wise strength, step gating) documented in consistent_self_attn.py and pipeline.py unless the task explicitly changes them.
4. If you are touching batch generation or consistency logic also read .opencode/project_state.md for the latest regression context.
