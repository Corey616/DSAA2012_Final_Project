# Prompt Issues Analysis — All 32 Stories

## Identity Core Issues (pre-Rule 13)

| Story | Character | identity_core items | Issue |
|-------|-----------|-------------------|-------|
| 06 | Jack | 2 | Missing skin tone, age, distinguishing features |
| 06 | Sara | 2 | Missing skin tone, age, distinguishing features |
| 11 | Olivia | 2 | "slender build, fair skin" — no hair/eye color |
| 13 | Robot | 2 | Good description but only 2 items |
| extra_05 | sam | 2 | "medium build, brown eyes" — too vague |
| extra_10 | woman | 2 | "warm skin, hair in bun" — missing eye color, face shape |

## Animal/Non-Human Identity Issues

| Story | Problem |
|-------|---------|
| 03 (cat) | "cat, fur, glossy pelt" — generic, no breed/color specifics |
| 03 (dog) | "dog, mottled tan dense woolly coat" — better but only 2 items |
| extra_06 (bird) | "Bird, rust colored smooth feathers" — decent, consistent |
| extra_03 (dog) | "Dog, golden brown thick fur" — adequate but minimal |

## Over-Saturation (SCA-specific)

17/32 SCA stories had saturation > 0.35. Fixed by reducing guidance 4.0 → 3.5.

## Recommendations

### LLM Parser (Rule 13 — already added)
- 4-5 distinctive features per character
- Human: hair color+texture, eye color, skin tone, face shape, age
- Non-human: species coloring, body type, texture, markings

### DiT Parameter Tuning
| Parameter | Current | Suggested Range | Impact |
|-----------|---------|----------------|--------|
| consistency_strength | 0.20 | 0.15-0.25 | Frame consistency vs detail |
| guidance_scale | 3.5 | 3.0-4.0 | Saturation vs prompt adherence |
| num_steps | 30 | 25-35 | Quality vs speed |
| sca_window_size | 1 | 1-2 | Cross-frame context |
| Late layer SCA | disabled | disabled | Preserves fine details |

### Batch Runner Improvements
- All 6 GPUs in parallel for full 32-story batch
- Each GPU handles 5-6 stories
- Runtime ~2-3 minutes per story → ~12 min total
