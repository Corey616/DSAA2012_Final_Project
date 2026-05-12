#!/usr/bin/env python3
"""Quick smoke test for HunyuanDiT pipeline."""
import sys
import os
import numpy as np

sys.path.insert(0, '.')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.hunyuan_pipeline import HunyuanGenerationPipeline
import torch
import gc

parser = create_qwen_parser(device_map='cuda:0')
with parser:
    board = parser.process_script_file('data/TaskA/11.txt')
del parser
torch.cuda.empty_cache()
gc.collect()

config = {
    'device': 'cuda:0',
    'use_fp16': True,
    'consistency_strength': 0.0,
    'generation_params': {
        'num_steps': 20,
        'guidance_scale': 4.5,
    },
}
pipe = HunyuanGenerationPipeline(config)
images, _ = pipe.generate_story(board, seed=42)
del pipe
torch.cuda.empty_cache()
gc.collect()

arrs = [np.array(im) for im in images]
for i, a in enumerate(arrs):
    print(f'Frame {i+1}: mean={a.mean():.0f} std={a.std():.0f}')
print('HUNYUAN SMOKE PASSED')
