# Skill: experiment-designer

## Purpose
Design formal experiments with explicit hypotheses, controlled conditions, and measurable success criteria before implementation begins. Prevents non-conclusive experiments that waste iterations.

## When to use
- Before implementing any code change that affects generation quality
- When the orchestrator needs to prioritize between multiple possible experiments
- When an experiment needs to be registered in the experiment registry

## Workflow

### 1. Read Context
Read the current experiment registry and project state:
- Read `.opencode/experiment_registry.yaml`
- Read `.opencode/project_state.md`
- Identify the highest-priority experiment that is READY (not blocked by dependencies)

### 2. Design Experiment
For each experiment, specify:

```yaml
experiment:
  id: "unique_experiment_id"
  hypothesis: "Clear falsifiable statement about what we expect to happen"
  
  control:
    variant: "hunyuan_multi" | "hunyuan_nosca" | "hunyuan_sca"
    stories: ["03", "05", ...]  # Must specify exact test set
    
  conditions:
    - name: "treatment_A"
      config: "HUNYUAN_SCA=1 TM=1 POSMASK=cross_attn"
      expected_effect: "Consistency +0.05, CLIP -0.01"
    - name: "treatment_B"
      config: "HUNYUAN_SCA=1 TM=1 POSMASK=off"
      
  success_criteria:
    - "Primary metric: Overall >= 0.45 on test set"
    - "Secondary: No story regresses by > 0.02"
    - "Tertiary: vj_count_drift <= 2 on all test stories"
    
  priority: "P0" | "P1" | "P2" | "P3"
  depends_on: []  # List of experiment IDs that must complete first
```

### 3. Register in Registry
Write the experiment to the registry file.

### 4. Verify Criteria Are Measurable
Each success criterion must be:
- **Measurable**: Can be computed from frame_clip_scores.json or evaluation.json
- **Binary**: Pass/fail, not "we think it looks better"
- **Realistic**: Within achievable range (not asking for CLIP=0.50 on a hard story)

### 5. Quality Checklist
- [ ] Hypothesis is falsifiable
- [ ] Control is specified (exact variant + test set)
- [ ] Success criteria are numeric and measurable
- [ ] Success criteria are prioritized (primary/secondary/tertiary)
- [ ] Dependencies are identified
- [ ] Test stories include at least 6 varied cases

## Example Experiment Design

```yaml
experiment:
  id: "cross_attn_posmask_v1"
  hypothesis: "Cross-attention map fallback allows position masks to activate on stories without spatial keywords, reducing multi-character feature leakage"
  control:
    variant: "hunyuan_multi"
    stories: ["01", "02", "03", "09", "17", "07"]
  conditions:
    - name: "cross_attn_posmask"
      config: "HUNYUAN_SCA=1, HUNYUAN_NO_TM=0"
  success_criteria:
    primary: "Position masks activate on >= 4/5 spatial-test stories"
    secondary: "vj_count_drift on 03: 5 -> <= 2"
    tertiary: "CLIP regression on 07: <= 0.01"
  priority: "P0"
  depends_on: []
```

## Output
Write the experiment design to the conversation for orchestrator review, then update the registry file.
