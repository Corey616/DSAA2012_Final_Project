#!/usr/bin/env python3
"""HunyuanDiT full batch runner. Processes all data/TaskA/*.txt stories."""
import sys, os, gc, torch, numpy as np, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors(); configure_all_cache_dirs()
from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.hunyuan_pipeline import HunyuanGenerationPipeline
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator

SCA = bool(os.environ.get('HUNYUAN_SCA', '0') == '1')
LABEL = 'hunyuan_sca' if SCA else 'hunyuan_nosca'
OUTPUT = Path(f'outputs/{LABEL}')
OUTPUT.mkdir(parents=True, exist_ok=True)
CLIP_EVAL = CLIPEvaluator()
CONSIST_EVAL = ConsistencyEvaluator()

# Parse --stories argument
parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--stories', type=str, default=None,
                        help='Space-separated story IDs to process, e.g. "01 02 03"')
args = parser_arg.parse_args()

DATA = Path('data/TaskA')
parser = create_qwen_parser(device_map='cuda:0')
results = []

# If --stories given, only process those; otherwise all *.txt sorted
if args.stories:
    story_ids = args.stories.strip().split()
    script_paths = []
    for sid in story_ids:
        sp = DATA / f'{sid}.txt'
        if sp.exists():
            script_paths.append(sp)
        else:
            print(f'WARNING: story {sid} not found at {sp}', flush=True)
else:
    script_paths = sorted(DATA.glob('*.txt'))

for script_path in script_paths:
    sid = script_path.stem
    print(f'\n=== {sid} ===', flush=True)
    try:
        with parser: board = parser.process_script_file(str(script_path))
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  Parse failed: {e}')
        results.append({'script':sid,'error':str(e)})
        continue

    config = {
        'device': 'cuda:0', 'use_fp16': True,
        'consistency_strength': 0.20 if SCA else 0.0,
        'sca_window_size': 1,
        'generation_params': {'num_steps': 30, 'guidance_scale': 3.5},
        'height': 1024, 'width': 1024,
    }
    pipe = HunyuanGenerationPipeline(config)
    try:
        images, plan = pipe.generate_story(board, seed=42)
        story_dir = OUTPUT / sid
        story_dir.mkdir(exist_ok=True)
        # Save images
        for i, img in enumerate(images):
            img.save(str(story_dir / f'frame_{i+1:02d}.png'))
        # Use compiled prompts (what model actually saw)
        prompts = []
        if plan:
            for p in plan:
                cp = p.get('compiled_prompt', '') or p.get('enhanced_prompt', '')
                prompts.append(cp)
        if not prompts:
            for i, panel in enumerate(board.panels):
                p = panel.enhanced_prompt or panel.raw_prompt or ""
                prompts.append(p)
        # Evaluate CLIP
        clip_scores = CLIP_EVAL.compute_similarity(images, prompts) if prompts and images else [0.0]*len(images)
        avg_clip = float(np.mean(clip_scores)) if clip_scores else 0.0
        # Evaluate consistency
        if len(images) >= 2:
            consist_dict = CONSIST_EVAL.compute_pairwise_consistency(images)
            avg_lpips = consist_dict.get('average_lpips', 0.5)
            consistency = 1.0 - avg_lpips
        else:
            consistency = 0.0
        overall = 0.6 * avg_clip + 0.4 * consistency
        print(f'{sid}: CLIP={avg_clip:.3f} Consist={consistency:.3f} (raw_lpips={1.0-consistency:.3f}) Overall={overall:.3f}', flush=True)
        results.append({
            'script': sid,
            'avg_clip_score': avg_clip,
            'avg_consistency': consistency,
            'raw_lpips': 1.0 - consistency,
            'overall_score': overall,
            'num_frames': len(images),
        })
        # Save individual frame CLIP scores for analysis
        frame_scores = {f'frame_{i+1:02d}': s for i, s in enumerate(clip_scores)}
        (story_dir / 'frame_clip_scores.json').write_text(
            json.dumps(frame_scores, indent=2)
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  Gen failed: {e}')
        results.append({'script': sid, 'error': str(e)})
    finally:
        del pipe; torch.cuda.empty_cache(); gc.collect()

# Save summary
summary = {
    'num_stories': len(results),
    'sca': SCA,
    'label': LABEL,
    'results': results,
}
(OUTPUT / 'batch_summary.json').write_text(json.dumps(summary, indent=2))
clips = [r['avg_clip_score'] for r in results if 'error' not in r]
conss = [r['avg_consistency'] for r in results if 'error' not in r]
if clips:
    print(f'\n{"="*60}')
    print(f'{LABEL} ({len(clips)}/{len(results)} stories succeeded):')
    print(f'  CLIP Score:       {np.mean(clips):.3f} ± {np.std(clips):.3f}')
    print(f'  Consistency:      {np.mean(conss):.3f} ± {np.std(conss):.3f}')
    overalls = [0.6*c + 0.4*co for c, co in zip(clips, conss)]
    print(f'  Overall Score:    {np.mean(overalls):.3f} ± {np.std(overalls):.3f}')
    print(f'{"="*60}')
failures = [r for r in results if 'error' in r]
if failures:
    print(f'\nFailures ({len(failures)}):')
    for f in failures:
        print(f'  {f["script"]}: {f["error"][:120]}')
