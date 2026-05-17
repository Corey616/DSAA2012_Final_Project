---
description: Remote execution on lzz@10.120.20.219 (8x A800 80GB). Sync code+models, run batch, retrieve results.
compatibility: ssh, rsync, conda
---

# Remote Execution Skill

## Quick Sync (code only)
rsync -avz --exclude outputs/ --exclude storygen/models/ --exclude __pycache__/ --exclude .git/ ./ lzz@10.120.20.219:~/DSAA2012FinalNew/

## Remote Run (smoke)
ssh lzz@10.120.20.219 "cd ~/DSAA2012FinalNew && CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sd35_v1.py --stories STORIES"

## Remote Run (full)
ssh lzz@10.120.20.219 "cd ~/DSAA2012FinalNew && CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_sd35_v1.py"

## Retrieve Results
rsync -avz lzz@10.120.20.219:~/DSAA2012FinalNew/outputs/ ./outputs_remote/

## Setup (first time)
ssh lzz@10.120.20.219 "conda create -n storygen python=3.12 -y && conda run -n storygen pip install diffusers transformers torch torchvision open_clip_torch pillow numpy"
