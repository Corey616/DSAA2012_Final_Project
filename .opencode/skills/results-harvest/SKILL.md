---
description: Fetch experiment results from remote A800, compare with local baselines, generate diff reports.
compatibility: rsync, ssh
---

# Results Harvest

## Fetch all results
rsync -avz lzz@10.120.20.219:~/DSAA2012FinalNew/outputs/ ./outputs_remote/

## Fetch specific experiment
rsync -avz lzz@10.120.20.219:~/DSAA2012FinalNew/outputs/batch_summary.json ./outputs_remote/

## Compare with local baseline
python3 -c "
import json, numpy as np
local = json.load(open('outputs/sd35_v1/batch_summary.json'))
remote = json.load(open('outputs_remote/batch_summary.json'))
lr = [r['dino_identity'] for r in local['results'] if 'error' not in r]
rr = [r['dino_identity'] for r in remote['results'] if 'error' not in r]
print(f'Local DINO: {np.mean(lr):.3f}  Remote DINO: {np.mean(rr):.3f}')
"
