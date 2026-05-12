#!/usr/bin/env python3
"""HunyuanDiT generation-only. Loads ProductionBoard from pickle."""
import sys, os, gc, torch, numpy as np, argparse, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors(); configure_all_cache_dirs()
from storygen.core_generator.hunyuan_pipeline import HunyuanGenerationPipeline
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator

SCA = bool(os.environ.get('HUNYUAN_SCA', '0') == '1')
LABEL = 'hunyuan_sca' if SCA else 'hunyuan_nosca'
OUTPUT = Path(f'outputs/{LABEL}')
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('outputs/parsed_boards_pkl')

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--stories', type=str, default=None)
args = parser_arg.parse_args()

CLIP_EVAL = CLIPEvaluator()
CONSIST_EVAL = ConsistencyEvaluator()
results = []

if args.stories:
    story_ids = args.stories.strip().split()
else:
    story_ids = sorted([f.stem for f in CACHE.glob('*.pkl')])

for sid in story_ids:
    board_path = CACHE / f'{sid}.pkl'
    if not board_path.exists():
        print(f'WARNING: parsed board {sid} not found', flush=True)
        continue
    board = pickle.load(open(board_path, 'rb'))
    print(f'\n=== {sid} ({len(board.panels)} panels) ===', flush=True)

    config = {
        'device': 'cuda:0', 'use_fp16': True,
        'consistency_strength': 0.6 if SCA else 0.0,
        'sca_window_size': 1,
        'generation_params': {'num_steps': 20, 'guidance_scale': 5.0},
        'height': 1024, 'width': 1024,
    }
    pipe = HunyuanGenerationPipeline(config)
    try:
        images, plan = pipe.generate_story(board, seed=42)
        story_dir = OUTPUT / sid
        story_dir.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            img.save(str(story_dir / f'frame_{i+1:02d}.png'))
        prompts = []
        if plan:
            for p in plan:
                cp = p.get('compiled_prompt', '') or p.get('enhanced_prompt', '')
                prompts.append(cp)
        if not prompts:
            for i, p in enumerate(board.panels):
                prompts.append(p.enhanced_prompt or p.raw_prompt or '')
        clip_scores = CLIP_EVAL.compute_similarity(images, prompts) if prompts and images else [0.0]*len(images)
        avg_clip = float(np.mean(clip_scores)) if clip_scores else 0.0
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
        frame_scores = {f'frame_{i+1:02d}': s for i, s in enumerate(clip_scores)}
        (story_dir / 'frame_clip_scores.json').write_text(json.dumps(frame_scores, indent=2))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  Gen failed: {e}')
        results.append({'script': sid, 'error': str(e)})
    finally:
        del pipe; torch.cuda.empty_cache(); gc.collect()

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
