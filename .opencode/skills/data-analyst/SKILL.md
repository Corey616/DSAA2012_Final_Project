# Skill: data-analyst

## Purpose
Generate clean comparison tables across experiment runs. No interpretation, no verdict — just tables.

## When to use
After tester runs complete and frame_clip_scores.json exists.

## Workflow

### 1. Read Data
Read frame_clip_scores.json from each variant directory.
Compute pixel consistency from frame PNGs.

### 2. Generate Table

| Story | Base CLIP | Cand CLIP | Delta | Base Cons | Cand Cons | Delta | Base Over | Cand Over | Delta |

### 3. Compute Summary
Mean, std, improvement/regression counts.

### 4. Output to Experiment-Reviewer
Experiment-reviewer (pro) will interpret.

## Example
python3 -c "
import json,os,numpy as np
from PIL import Image
def pc(b,s):
  imgs=[]
  for i in range(3):
    f=os.path.join(b,s,f'frame_0{i+1}.png')
    if os.path.exists(f): imgs.append(np.array(Image.open(f)))
  if len(imgs)<2: return 0.5
  ds=[np.abs(imgs[i].astype(float)-imgs[i+1].astype(float)).mean() for i in range(len(imgs)-1)]
  return max(0,1.0-sum(ds)/len(ds)/200)
def clip(b,s):
  f=os.path.join(b,s,'frame_clip_scores.json')
  if not os.path.exists(f): return None
  return np.mean(list(json.load(open(f)).values()))
"

## Output
Raw comparison tables. No verdict.
