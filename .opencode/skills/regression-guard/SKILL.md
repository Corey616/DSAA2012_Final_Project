# Skill: regression-guard

## Purpose
Automatically detect metric regressions across iterations. Monitors frame_clip_scores.json, evaluation.json, and batch_summary.json for statistically significant drops.

## When to use
- After every tester run that produces new frame_clip_scores.json
- Before committing code changes (as a pre-commit gate)
- When the orchestrator needs to decide PROCEED vs ROLLBACK

## Workflow

### 1. Collect Baseline
Read the most recent trusted batch_summary.json (from outputs/taskA_batch/ or outputs/hunyuan_multi/).

### 2. Collect Candidate
Read the new run's batch_summary.json (from outputs/hunyuan_sca/ or outputs/refactored_hunyuan_sca/).

### 3. Per-Story Comparison
For each story in both runs, compare:

```
| Story | Base CLIP | Cand CLIP | Delta | Base Cons | Cand Cons | Delta |
|-------|-----------|-----------|-------|-----------|-----------|-------|
| 03    | 0.255     | 0.237     | -0.018| 0.656     | 0.756     | +0.100|
```

### 4. Flag Regressions
A regression is flagged when:
- **Critical**: Any story drops > 0.02 overall (== 2x baseline noise floor)
- **Warning**: Any story drops > 0.01 overall
- **Info**: Any story drops > 0.005 overall
- **Improvement**: Any story improves > 0.01 overall

### 5. Generate Report

```
=== REGRESSION GUARD REPORT ===
Run: cross_attn_posmask_v1 vs baseline (hunyuan_multi)
Date: 2026-05-14

CRITICAL REGRESSIONS (drop > 0.02):
  None ✓

WARNINGS (drop 0.01-0.02):
  extra_07: overall 0.472 -> 0.445 (-0.027) [CLIP -0.054, Cons +0.006]
  Reason: TM over-smoothing on high-CLIP story

IMPROVEMENTS (gain > 0.01):
  03: overall 0.378 -> 0.444 (+0.067) [CLIP +0.035, Cons +0.114]
  05: overall 0.423 -> 0.487 (+0.064) [CLIP +0.009, Cons +0.147]
  13: overall 0.391 -> 0.450 (+0.059) [CLIP +0.004, Cons +0.141]

VERDICT: PROCEED with caution (2 regressions, 3 improvements)
```

### 6. Regression Patterns Reference

| Pattern | Typical Delta | Likely Root Cause |
|---------|-------------|-------------------|
| CLIP down + Cons up | -0.02 CLIP, +0.10 Cons | Token merging over-smoothing |
| CLIP down + Cons down | -0.03 both | Attention mask blocking useful features |
| CLIP stable + Cons down | -0.05 Cons | SCA K/V concat feature averaging |
| VLM tag increase | vj_* tags +2 | Multi-character leakage (position masks needed) |

## Confidence Levels
- **High confidence**: >= 6 stories tested, < 0.005 noise floor
- **Medium confidence**: 3-5 stories tested, < 0.01 noise floor  
- **Low confidence**: < 3 stories tested, report as exploratory

## Output
Write regression report to conversation for orchestrator and experiment-reviewer.
