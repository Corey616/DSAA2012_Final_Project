---
description: Export/import conda environments between local and remote (lzz@10.120.20.219). Handles version conflicts.
compatibility: conda, ssh
---

# Conda Env Sync

## Export local env
conda env export -n base > environment.yaml

## Create on remote (first time)
scp environment.yaml lzz@10.120.20.219:~/DSAA2012FinalNew/
ssh lzz@10.120.20.219 "conda env create -f ~/DSAA2012FinalNew/environment.yaml -n storygen"

## Install missing packages on remote
ssh lzz@10.120.20.219 "source ~/miniforge3/etc/profile.d/conda.sh && conda activate storygen && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>"

## Verify
ssh lzz@10.120.20.219 "source ~/miniforge3/etc/profile.d/conda.sh && conda activate storygen && python3 -c 'import diffusers, torch; print("OK")'"
