#!/usr/bin/env python3
"""Parallel HunyuanDiT batch runner — dispatches 32 stories across 6 GPUs."""
import subprocess, sys, os, gc, torch, json, time
from pathlib import Path

# Config
SCA = bool(int(os.environ.get('HUNYUAN_SCA', '0')))
LABEL = 'hunyuan_sca' if SCA else 'hunyuan_nosca'
OUTPUT = Path(f'outputs/{LABEL}')
GPUS = [2,3,4,5,6,7]  # Available GPUs
DATA = sorted(Path('data/TaskA').glob('*.txt'))

print(f'=== HunyuanDiT Parallel Batch ({LABEL}) ===')
print(f'Stories: {len(DATA)}, GPUs: {GPUS}')

# Assign stories to GPUs (round-robin)
gpu_assignments = {g: [] for g in GPUS}
for i, script in enumerate(DATA):
    gpu = GPUS[i % len(GPUS)]
    gpu_assignments[gpu].append(script.stem)

procs = []
for gpu, stories in gpu_assignments.items():
    stories_str = ' '.join(stories)
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} HUNYUAN_SCA={int(SCA)} python3 scripts/run_hunyuan_batch.py --stories {stories_str}'
    print(f'  GPU {gpu}: {len(stories)} stories')
    log = OUTPUT / f'gpu_{gpu}.log'
    p = subprocess.Popen(cmd.split(), stdout=open(log,'w'), stderr=subprocess.STDOUT)
    procs.append((gpu, p))

# Wait for all
for gpu, p in procs:
    p.wait()
    print(f'  GPU {gpu}: done (rc={p.returncode})')

# Aggregate results
import glob
results = []
for subdir in sorted(OUTPUT.iterdir()):
    if not subdir.is_dir(): continue
    bj = subdir / 'batch_summary.json'
    if bj.exists():
        with open(bj) as f:
            r = json.load(f)
        results.extend(r.get('results',r) if isinstance(r,dict) else r)
    else:
        # Check for individual story results
        ej = subdir / 'evaluation.json'
        if ej.exists():
            with open(ej) as f:
                results.append(json.load(f))

if results:
    valid = [r for r in results if 'error' not in r]
    clips = [r['avg_clip_score'] for r in valid]
    conss = [r['avg_consistency'] for r in valid]
    print(f'\n{LABEL} FINAL ({len(valid)}/{len(results)} valid):')
    print(f'  CLIP: {sum(clips)/len(clips):.3f}  Consist: {sum(conss)/len(conss):.3f}  Overall: {sum(0.6*c+0.4*co for c,co in zip(clips,conss))/len(clips):.3f}')
