#!/usr/bin/env python3
"""
Parallel Batch Generator — uses multiple GPUs simultaneously.
Each story runs on its own GPU via subprocess.
"""
import subprocess
import sys
import time
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from storygen.core_generator.config_defaults import build_generation_config

# Stories that need generation (extra stories have no frames)
EXTRA_STORIES = [f"extra_{i:02d}" for i in range(1, 13)]

# GPU memory requirements (MB)
SDXL_GEN = 20000  # Full SDXL generation
QWN_PARSE = 2048  # Qwen parsing

def get_free_gpus(min_mem_mb=SDXL_GEN + QWN_PARSE):
    """Get sorted list of (gpu_id, free_mem) for GPUs with enough free memory."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    free = []
    for line in result.stdout.strip().split("\n"):
        idx, mem = line.split(", ")
        idx, mem = int(idx), int(mem)
        if mem >= min_mem_mb:
            free.append((idx, mem))
    return sorted(free, key=lambda x: -x[1])

def run_story_on_gpu(story_id: str, gpu_id: int, story_idx: int):
    """Run batch processing for a single story on a specific GPU."""
    script_file = f"data/TaskA/{story_id}.txt"
    output_dir = str(BASE / "storygen" / "outputs" / "taskA_batch" / story_id)
    config_json = json.dumps(build_generation_config(device="cuda:0"))

    cmd = [
        sys.executable, "-c", f"""
import sys, os
sys.path.insert(0, '{BASE}')
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

import torch
from pathlib import Path
import json
from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.pipeline import NarrativeGenerationPipeline

script_file = '{script_file}'
output_dir = Path('{output_dir}')
output_dir.mkdir(parents=True, exist_ok=True)

# Parse
parser = create_qwen_parser(device_map='cuda:0')
with parser:
    board = parser.process_script_file(script_file)
    parser.save_production_board(board, str(output_dir / 'production_board.json'))
del parser
torch.cuda.empty_cache()

# Generate
config = json.loads('''{config_json}''')
pipe = NarrativeGenerationPipeline(config)
images, _ = pipe.generate_story(board, seed=42)

from storygen.utils.image_utils import remove_white_borders
for i, img in enumerate(images, 1):
    clean = remove_white_borders(img)
    clean.save(output_dir / f'frame_{{i:02d}}.png')

# Quick eval
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator

clip_eval = CLIPEvaluator(device='cuda:0')
clip_scores = []
for i, panel in enumerate(board.panels):
    prompt = panel.enhanced_prompt or panel.raw_prompt
    score = clip_eval.compute_similarity([images[i]], [prompt])[0]
    clip_scores.append(score)
avg_clip = sum(clip_scores) / len(clip_scores)

if len(images) > 1:
    cons_eval = ConsistencyEvaluator(device='cuda:0', metric='lpips')
    lpips_scores = []
    for i in range(len(images) - 1):
        dist = cons_eval.compute_lpips_similarity(images[i], images[i + 1])
        lpips_scores.append(1 - dist)
    avg_cons = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 1.0
else:
    avg_cons = 1.0

overall = 0.6 * avg_clip + 0.4 * avg_cons
metrics = {{
    'script': script_file, 'num_panels': len(board.panels),
    'avg_clip_score': avg_clip, 'avg_consistency': avg_cons,
    'overall_score': overall,
}}
import json
with open(output_dir / 'evaluation.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'[{gpu_id}] {{story_id}}: CLIP={{avg_clip:.3f}} Consist={{avg_cons:.3f}} Overall={{overall:.3f}}')
"""
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return result.stdout + result.stderr


if __name__ == "__main__":
    print("=" * 60)
    print("Parallel Batch Generator — Extra Stories")
    print("=" * 60)
    
    free_gpus = get_free_gpus()
    print(f"\nFree GPUs with {SDXL_GEN + QWN_PARSE}MB+: {[g for g, _ in free_gpus]}")
    
    if not free_gpus:
        print("No GPUs available! Waiting...")
        while not free_gpus:
            time.sleep(30)
            free_gpus = get_free_gpus()
    
    # Distribute stories across GPUs
    gpu_count = min(len(free_gpus), len(EXTRA_STORIES))
    if gpu_count > 4:
        gpu_count = 4  # Limit to 4 GPUs max for stability
    
    # Assign stories to GPUs round-robin
    gpu_assignments = {gpu: [] for gpu, _ in free_gpus[:gpu_count]}
    for i, story in enumerate(EXTRA_STORIES):
        gpu = list(gpu_assignments.keys())[i % gpu_count]
        gpu_assignments[gpu].append(story)
    
    print(f"\nUsing {gpu_count} GPUs:")
    for gpu, stories in gpu_assignments.items():
        print(f"  GPU {gpu}: {stories}")
    
    # Launch processes
    processes = []
    for gpu, stories in gpu_assignments.items():
        for idx, story in enumerate(stories):
            print(f"\n[{gpu}] Starting {story}...")
            p = subprocess.Popen(
                [sys.executable, __file__, story, str(gpu), str(idx)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True
            )
            processes.append((story, gpu, p))
    
    # Wait and collect results
    results = []
    for story, gpu, p in processes:
        stdout, _ = p.communicate()
        print(stdout[-200:] if len(stdout) > 200 else stdout)
        results.append((story, gpu, p.returncode))
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for story, gpu, rc in results:
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  GPU {gpu}: {story} → {status}")
