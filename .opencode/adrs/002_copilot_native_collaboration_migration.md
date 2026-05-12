# ADR 002: Agent definition source of truth — SUPERSEDED

## Status: SUPERSEDED (2026-05-11)

The `.opencode/agents/*.md` (agent-forge format) is the sole source of truth for agent definitions.
`AGENTS.md` has been deleted to eliminate hardcoded architecture descriptions that cause hallucination risk.
See `.opencode/agents/orchestrator.md` for the current loop controller definition.
