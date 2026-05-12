---
name: debug-python
description: >-
  Standard debugging workflow for Python/storygen errors: (1) reproduce with
  minimal input, (2) isolate to specific module, (3) check CUDA OOM patterns,
  (4) log hidden states dimensions for attention mismatch, (5) verify config
  parameters. Covers SDXL pipeline debug patterns, attention dimension errors,
  and HF model loading issues.
license: MIT
compatibility:
  - python >= 3.10
metadata:
  priority: medium
  last_updated: 2026-05-02
---

# Python Debugging Workflow

## Step 1: Reproduce with Minimal Input
```bash
python quick_test.py --story storygen/data/TaskA/11.txt --verbose
```
Check if error is specific to certain stories or all stories.

## Step 2: Isolate Module
- **LLM parsing error** → check `llm_parser.py` output `ProductionBoard`
- **Generation error** → test with `test_basic_sdxl.py` bypassing custom processor
- **Attention error** → check dimension mismatch in `consistent_self_attn.py`
- **Evaluation error** → run `metric_clip.py` or `metric_consistency.py` separately

## Common Error Patterns

### CUDA OOM
```python
# Solution: reduce batch size, enable CPU offload
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
```

### Attention Dimension Mismatch
```python
# Check dimensions:
# query: (batch, seq_len, dim) - dim = UNet hidden dim
# key/value must match query's dim after projection
# For SCA K/V concat, historical features must be projected to same dim
```

### HF Model Loading
```python
# Check mirror_config.py for cache directory
# Use: python cleanup_cache.py to remove incomplete downloads
```

## Step 3: Add Debug Logging
Insert dimension prints in attention processor:
```python
print(f"[DEBUG] Q shape: {query.shape}, K shape: {key.shape}, V shape: {value.shape}")
```

## Step 4: Minimal Reproduction
Always create a minimal test script (like `simple_test.py`) before integration.
