---
name: git-workflow
description: >-
  Git branch management, commit conventions, and PR workflow for this project.
  Branch naming: feature/<name>, fix/<name>. Commit messages: imperative mood,
  prefix with [SCA], [FACE], [LORA], [EVAL], or [CONFIG]. Never commit to
  main directly. Always run eval before committing changes.
license: MIT
metadata:
  priority: medium
  last_updated: 2026-05-02
---

# Git Workflow

## Branch Naming
- `feature/sca-impl` — SCA + batch generation
- `feature/face-lock` — Face identity embedding
- `feature/lora-nonhuman` — Non-human LoRA support
- `fix/<issue-description>` — Bug fixes
- `chore/<task>` — Config, docs, cleanup

## Commit Message Convention
```
[<TAG>] Imperative description in < 50 chars

Optional body explaining motivation and key changes.
```

Tags: `SCA`, `FACE`, `LORA`, `EVAL`, `CONFIG`, `FIX`, `DOCS`, `REFACTOR`

## Pre-Commit Checklist
1. Run `python run_taska_batch.py` to confirm no regressions
2. Compare against baseline metrics (CLIP 0.288 / Consistency 0.386)
3. If metrics drop, debug before committing
4. Only commit when explicitly asked

## PR Workflow
1. Branch from `main`
2. Implement changes
3. Run full eval
4. Create PR with summary of CLIP/Consistency impact
5. Do NOT push to main without review
