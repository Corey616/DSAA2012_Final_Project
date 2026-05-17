---
description: Selective model sync to remote A800. Syncs only SD3.5 + Qwen (~30GB), skips unused models.
compatibility: rsync, ssh
---

# Model Sync

## Sync essential models only (SD3.5 + Qwen)
rsync -avzP --progress storygen/models/models--adamo1139--stable-diffusion-3.5-medium-ungated/ lzz@10.120.20.219:~/DSAA2012FinalNew/storygen/models/models--adamo1139--stable-diffusion-3.5-medium-ungated/

rsync -avzP --progress storygen/models/models--Qwen--Qwen2.5-7B-Instruct/ lzz@10.120.20.219:~/DSAA2012FinalNew/storygen/models/models--Qwen--Qwen2.5-7B-Instruct/

## Verify model files
ssh lzz@10.120.20.219 "ls ~/DSAA2012FinalNew/storygen/models/ | head -5"

## Sync all (128GB, slow)
nohup rsync -avzP storygen/models/ lzz@10.120.20.219:~/DSAA2012FinalNew/storygen/models/ &
