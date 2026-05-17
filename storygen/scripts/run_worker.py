#!/usr/bin/env python3
"""
Single-GPU worker for parallel batch execution.
Usage: python run_worker.py --gpu 0 --script data/TaskA/01.txt --output outputs/parallel/01
"""
import sys
import os
from pathlib import Path
# Ensure CWD (storygen/) is in sys.path for 'from src.xxx' imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys
import os
import json
import argparse
from pathlib import Path

# CWD is storygen/ — already in sys.path


def run_single(gpu_id: int, script_file: str, output_dir: str):
    """Run a single test case on a specific GPU. Saves results to JSON."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from src.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
    setup_china_mirrors()
    configure_all_cache_dirs()

    import torch
    import gc
    import time
    from PIL import Image
    import traceback

    from src.script_director.llm_parser_local import create_qwen_parser
    from src.core_generator.pipeline import NarrativeGenerationPipeline
    from src.evaluation_hub.metric_clip import CLIPEvaluator
    from src.evaluation_hub.metric_consistency import ConsistencyEvaluator
    from src.utils.image_utils import create_storyboard, remove_white_borders

    script_path = Path(script_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "script": str(script_file),
        "gpu": gpu_id,
        "status": "pending",
    }

    try:
        t0 = time.time()
        parser = create_qwen_parser()
        with parser:
            board = parser.process_script_file(script_file)
            parser.save_production_board(board, str(output_path / "production_board.json"))
        parse_time = time.time() - t0
        del parser
        torch.cuda.empty_cache()

        result["num_panels"] = len(board.panels)
        result["num_characters"] = len(board.characters)
        result["parse_time_s"] = round(parse_time, 1)

        # ── Step 2: Generate ──
        config = {
            "device": "cuda:0",  # CUDA_VISIBLE_DEVICES maps this to gpu_id
            "use_fp16": True,
            "consistency_strength": 0.0,
            "memory_bank_size": 4,
            "generation_params": {"num_steps": 40, "guidance_scale": 7.5},
            "height": 1024,
            "width": 1024,
            "enable_model_cpu_offload": True,
        }

        pipeline = NarrativeGenerationPipeline(config)
        t1 = time.time()
        images, _ = pipeline.generate_story(board, seed=42)
        gen_time = time.time() - t1
        result["generation_time_s"] = round(gen_time, 1)
        result["num_frames"] = len(images)

        # Save
        for i, img in enumerate(images, 1):
            img_clean = remove_white_borders(img)
            img_clean.save(output_path / f"frame_{i:02d}.png")

        images_clean = [remove_white_borders(img) for img in images]
        storyboard = create_storyboard(
            images_clean,
            [f"Scene {i+1}: {p.shot_type}" for i, p in enumerate(board.panels)],
            image_size=(512, 512),
        )
        storyboard.save(output_path / "storyboard.png")

        # ── Step 3: Evaluate ──
        clip_eval = CLIPEvaluator(device="cuda:0")
        clip_scores = []
        for i, panel in enumerate(board.panels):
            prompt = panel.enhanced_prompt or panel.raw_prompt
            img_clean = remove_white_borders(images[i])
            score = clip_eval.compute_similarity([img_clean], [prompt])[0]
            clip_scores.append(float(score))
        del clip_eval
        torch.cuda.empty_cache()

        avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0
        result["clip_scores"] = clip_scores
        result["avg_clip_score"] = round(avg_clip, 4)

        # Consistency
        if len(images_clean) > 1:
            cons_eval = ConsistencyEvaluator(device="cuda:0", metric="lpips")
            lpips_scores = []
            for i in range(len(images_clean) - 1):
                dist = cons_eval.compute_lpips_similarity(images_clean[i], images_clean[i + 1])
                lpips_scores.append(round(1.0 - dist, 4))
            avg_cons = sum(lpips_scores) / len(lpips_scores)
            del cons_eval
        else:
            avg_cons = 1.0
            lpips_scores = []

        result["lpips_scores"] = lpips_scores
        result["avg_consistency"] = round(avg_cons, 4)
        result["overall_score"] = round(0.6 * avg_clip + 0.4 * avg_cons, 4)
        result["status"] = "success"

        del pipeline
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    # Save individual result
    with open(output_path / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Print summary line for orchestrator to capture
    status_icon = "✅" if result["status"] == "success" else "❌"
    metric_str = ""
    if result["status"] == "success":
        metric_str = f"CLIP={result['avg_clip_score']:.3f} CONS={result['avg_consistency']:.3f} OVERALL={result['overall_score']:.3f}"
    print(f"WORKER_DONE|{status_icon}|GPU{gpu_id}|{script_path.name}|{result['status']}|{metric_str}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_single(args.gpu, args.script, args.output)
