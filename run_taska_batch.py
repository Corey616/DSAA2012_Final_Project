#!/usr/bin/env python3
"""
TaskA Full Test - Batch evaluation on all TaskA data files
GPU-aware: dynamically selects available GPUs, waits if none available.
Uses DeepSeek API for LLM-based story parsing (no local Qwen needed).
"""
import sys
import os
sys.path.insert(0, '.')

# Configure once at import
if not os.environ.get("STORYGEN_MIRRORS_CONFIGURED"):
    from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
    setup_china_mirrors()
    configure_all_cache_dirs()
    os.environ["STORYGEN_MIRRORS_CONFIGURED"] = "1"

import torch
from PIL import Image
from pathlib import Path
import json
import time
import gc
from datetime import datetime

# Use DeepSeek API for story parsing (no local Qwen needed)
from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.config_defaults import build_generation_config
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator
from storygen.utils.image_utils import create_storyboard, remove_white_borders

from scripts.gpu_scheduler import acquire_gpu, wait_for_gpu, release_gpu, release_all_gpus


def test_single_file(
    script_file: str,
    output_dir: str,
    device: str = None,
    wait: bool = False,
) -> dict:
    """Test a single TaskA file with dynamic GPU selection."""
    if device is None:
        if wait:
            device = f"cuda:{wait_for_gpu(8000, holder=f'sdxl_gen_{Path(script_file).stem}')}"
        else:
            gpu = acquire_gpu(8000, holder=f'sdxl_gen_{Path(script_file).stem}')
            if gpu is None:
                return {"script": str(script_file), "error": "No GPU available"}
            device = f"cuda:{gpu}"

    print(f"\n{'='*60}")
    print(f"Testing: {script_file} on {device}")
    print(f"{'='*60}")

    script_path = Path(script_file)
    output_path = Path(output_dir) / script_path.stem
    os.makedirs(output_path, exist_ok=True)

    # Step 1: Parse script using local Qwen (GPU-aware, will raise if no memory)
    start = time.time()
    parse_gpu = acquire_gpu(4096, holder=f"qwen_parse_{script_path.stem}")
    if parse_gpu is None:
        if not wait:
            return {"script": str(script_file), "error": "No GPU for Qwen parser"}
        parse_gpu = wait_for_gpu(4096, holder=f"qwen_parse_{script_path.stem}")

    # Use CUDA_VISIBLE_DEVICES to pin Qwen to specific GPU (4-bit model doesn't support manual device_map)
    qwen_env = os.environ.copy()
    qwen_env["CUDA_VISIBLE_DEVICES"] = str(parse_gpu)
    qwen_env["HF_HOME"] = str(Path(__file__).parent / "storygen/models")
    qwen_env["HF_HUB_CACHE"] = str(Path(__file__).parent / "storygen/models")
    # Spawn subprocess for Qwen parsing to avoid GPU conflicts
    import subprocess
    result = subprocess.run(
        ["python3", "-c", f"""
import sys
sys.path.insert(0, '.')
from storygen.script_director.llm_parser_local import create_qwen_parser
parser = create_qwen_parser()
with parser:
    board = parser.process_script_file("{script_file}")
board_data = {{
    "story_id": board.story_id, "global_style": board.global_style,
    "characters": {{k: v.__dict__ for k, v in board.characters.items()}},
    "panels": [p.__dict__ for p in board.panels],
    "consistency_constraints": board.consistency_constraints,
    "story_state": board.story_state,
    "render_plan": board.render_plan,
}}
import json
with open("{output_path / 'production_board.json'}", "w") as f:
    json.dump(board_data, f, indent=2, default=str)
print(f"Parsed: {{len(board.panels)}} panels, {{len(board.characters)}} characters")
"""],
        capture_output=True, text=True, timeout=300, env=qwen_env,
    )
    if result.returncode != 0:
        print(f"[Parser] Qwen parsing failed: {result.stderr[:200]}")
        release_gpu(parse_gpu)
        raise RuntimeError(f"Qwen parsing failed (exit={result.returncode})")
    print(result.stdout.strip())
    # Re-read board from saved file
    with open(output_path / "production_board.json") as f:
        board_data = json.load(f)
    from storygen.script_director.llm_parser import ProductionBoard, Character, Panel
    board = ProductionBoard(
        story_id=board_data["story_id"],
        characters={k: Character(**v) for k, v in board_data["characters"].items()},
        panels=[Panel(**p) for p in board_data["panels"]],
        global_style=board_data["global_style"],
        consistency_constraints=board_data.get("consistency_constraints", []),
        story_state=board_data.get("story_state", {}),
        render_plan=board_data.get("render_plan", []),
    )
    parse_time = time.time() - start
    del result
    torch.cuda.empty_cache()
    release_gpu(parse_gpu)

    print(f"  Parsed: {len(board.panels)} panels, {len(board.characters)} characters")

    # Step 2: Setup pipeline
    config = build_generation_config(
        {
            "enable_model_cpu_offload": True,
        },
        device=device,
    )

    pipeline = NarrativeGenerationPipeline(config)

    # Step 3: Generate
    gen_start = time.time()
    images, _ = pipeline.generate_story(board, seed=42)
    gen_time = time.time() - gen_start
    print(f"  Generated {len(images)} frames in {gen_time:.1f}s")

    # Save images
    for i, img in enumerate(images, 1):
        img_clean = remove_white_borders(img)
        img_clean.save(output_path / f"frame_{i:02d}.png")

    images_clean = [remove_white_borders(img) for img in images]
    storyboard = create_storyboard(
        images_clean,
        [f"Scene {i+1}: {p.shot_type}" for i, p in enumerate(board.panels)],
        image_size=(512, 512)
    )
    storyboard.save(output_path / "storyboard.png")

    # Release generation GPU before eval
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    gen_gpu_id = int(device.split(":")[1]) if ":" in device else -1
    release_gpu(gen_gpu_id)

    # Step 4: Evaluate (use any available GPU, different from generation)
    eval_gpu = acquire_gpu(2048, exclude_gpus={gen_gpu_id} if gen_gpu_id >= 0 else None,
                           holder=f"eval_{script_path.stem}")
    eval_device = f"cuda:{eval_gpu}" if eval_gpu is not None else "cpu"

    clip_eval = CLIPEvaluator(device=eval_device)
    clip_scores = []
    prompts = []
    for i, panel in enumerate(board.panels):
        if getattr(board, "render_plan", None) and i < len(board.render_plan):
            prompt = board.render_plan[i].get("compiled_prompt") or panel.enhanced_prompt or panel.raw_prompt
        else:
            prompt = panel.enhanced_prompt or panel.raw_prompt
        prompts.append(prompt)
        img_clean = remove_white_borders(images[i])
        score = clip_eval.compute_similarity([img_clean], [prompt])[0]
        clip_scores.append(score)
        print(f"  Frame {i+1} CLIP: {score:.3f} | {prompt[:50]}...")

    del clip_eval
    torch.cuda.empty_cache()

    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0

    images_clean = [remove_white_borders(img) for img in images]
    if len(images_clean) > 1:
        consistency_eval = ConsistencyEvaluator(device=eval_device, metric="lpips")
        lpips_scores = []
        for i in range(len(images_clean) - 1):
            dist = consistency_eval.compute_lpips_similarity(images_clean[i], images_clean[i + 1])
            lpips_scores.append(1 - dist)
        avg_consistency = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 1.0
        del consistency_eval
        torch.cuda.empty_cache()
    else:
        avg_consistency = 1.0

    if eval_gpu is not None:
        release_gpu(eval_gpu)

    overall = 0.6 * avg_clip + 0.4 * avg_consistency

    metrics = {
        "script": str(script_file),
        "num_panels": len(board.panels),
        "num_characters": len(board.characters),
        "prompts": prompts,
        "clip_scores": [float(s) for s in clip_scores],
        "avg_clip_score": float(avg_clip),
        "avg_consistency": float(avg_consistency),
        "overall_score": float(overall),
        "parse_time_s": round(parse_time, 1),
        "generation_time_s": round(gen_time, 1),
    }

    with open(output_path / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_path / "story_state.json", "w") as f:
        json.dump(getattr(board, "story_state", {}), f, indent=2)

    with open(output_path / "render_plan.json", "w") as f:
        json.dump(getattr(board, "render_plan", []), f, indent=2)

    with open(output_path / "production_board.json", "w") as f:
        json.dump({
            "story_id": board.story_id,
            "global_style": board.global_style,
            "characters": {k: v.__dict__ for k, v in board.characters.items()},
            "panels": [p.__dict__ for p in board.panels],
            "consistency_constraints": board.consistency_constraints,
            "story_state": getattr(board, "story_state", {}),
            "render_plan": getattr(board, "render_plan", []),
        }, f, indent=2, default=str)

    print(f"\n  Results: CLIP={avg_clip:.3f}, Consistency={avg_consistency:.3f}, Overall={overall:.3f}")
    gc.collect()

    return metrics


def run_taska_batch(
    output_dir: str = "storygen/outputs/taskA_batch",
    story_ids: list = None,
    wait: bool = False,
):
    """Run batch test on TaskA files with GPU scheduling."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║   TaskA Batch - GPU-Aware Evaluation                      ║
║   Dynamic GPU selection + wait-if-busy                    ║
╚═══════════════════════════════════════════════════════════╝
    """)

    taska_dir = Path("storygen/data/TaskA")
    taska_files = sorted(taska_dir.glob("*.txt"))
    if story_ids:
        taska_files = [f for f in taska_files if f.stem in story_ids]
    print(f"Found {len(taska_files)} files (wait={'yes' if wait else 'no'})")

    CLIP_THRESHOLD = 0.30
    CONSISTENCY_THRESHOLD = 0.30

    all_results = []
    os.makedirs(output_dir, exist_ok=True)

    for i, script_file in enumerate(taska_files, 1):
        print(f"\n[{i}/{len(taska_files)}]")
        try:
            metrics = test_single_file(
                str(script_file),
                output_dir=output_dir,
                wait=wait,
            )
            all_results.append(metrics)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "script": str(script_file),
                "error": str(e),
                "overall_score": 0.0
            })

    print("\n" + "="*70)
    print("TASKA BATCH TEST SUMMARY")
    print("="*70)

    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        avg_clip = sum(r["avg_clip_score"] for r in valid_results) / len(valid_results)
        avg_consistency = sum(r["avg_consistency"] for r in valid_results) / len(valid_results)
        avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)

        print(f"\nTotal: {len(taska_files)}, Successful: {len(valid_results)}, "
              f"Failed: {len(all_results) - len(valid_results)}")
        print(f"\n  Average CLIP:     {avg_clip:.3f} (threshold: {CLIP_THRESHOLD})")
        print(f"  Average Consist:  {avg_consistency:.3f} (threshold: {CONSISTENCY_THRESHOLD})")
        print(f"  Average Overall: {avg_overall:.3f}")

        for r in valid_results:
            status = "PASS" if r["overall_score"] >= 0.3 else "FAIL"
            print(f"  [{status}] {Path(r['script']).name}: "
                  f"CLIP={r['avg_clip_score']:.3f}, C={r['avg_consistency']:.3f}")
    else:
        print("  All files failed!")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(taska_files),
        "successful": len(valid_results),
        "results": all_results,
    }
    with open(Path(output_dir) / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    release_all_gpus()
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="Wait for GPU if busy")
    parser.add_argument("--story", nargs="+", help="Specific story IDs")
    args = parser.parse_args()
    run_taska_batch(story_ids=args.story, wait=args.wait)
