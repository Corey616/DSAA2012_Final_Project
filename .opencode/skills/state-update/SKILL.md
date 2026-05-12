---
name: state-update
description: >-
  Update StoryGen project state after a meaningful iteration. Use this when
  metrics blockers architecture decisions or durable lessons need to be written
  back to project files.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: high
  last_updated: 2026-05-08
---

Use this skill after evaluation or after a meaningful architectural change.

1. Update .opencode/project_state.md with the current phase blocker changes and the latest trusted metrics.
2. Update MEMORY.md only with durable facts or repeat failure patterns that should survive across sessions.
3. Add or extend an ADR in .opencode/adrs/ only for decisions that materially affect future implementation choices.
4. If MCP memory is configured mirror the same outcome through the memory server.
5. Do not mix rolling experiment logs into MEMORY.md.
