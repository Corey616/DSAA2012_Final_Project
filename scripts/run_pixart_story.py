"""
Run PixArt-Σ on a single story with LocalQwenParser and evaluation.
Usage: python scripts/run_pixart_story.py --story 04 [--no_sca]
"""
import argparse, json, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storygen.script_director.llm_parser_local import LocalQwenParser, create_qwen_parser
from storygen.core_generator.pipeline_pixart import PixArtGenerationPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", default="04")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--consistency", type=float, default=0.14)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--no_sca", action="store_true")
    parser.add_argument("--output_dir", default="outputs/pixart_results")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    script_path = Path(f"storygen/data/TaskA/{args.story}.txt")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cs = args.consistency if not args.no_sca else 0.0
    print(f"[Story {args.story}] PixArt-Σ with SCA={cs}, window={args.window}, steps={args.steps}")

    print("\n[Phase 1] LLM Parsing with Qwen2.5-7B...")
    t0 = time.time()
    parser = create_qwen_parser(device_map=f"cuda:{args.gpu}")
    with parser:
        board = parser.process_script_file(str(script_path))
    parse_time = time.time() - t0
    del parser
    torch.cuda.empty_cache()
    print(f"  Parse done: {parse_time:.1f}s, {len(board.panels)} panels")

    config = {
        "consistency_strength": cs,
        "sca_window_size": args.window,
        "sca_start_ratio": 0.25,
        "generation_params": {"num_steps": args.steps, "guidance_scale": args.guidance},
        "height": 1024, "width": 1024,
    }

    print("\n[Phase 2] PixArt-Σ Generation...")
    t1 = time.time()
    pipe = PixArtGenerationPipeline(config)
    images, _ = pipe.generate_story(board, seed=42)
    gen_time = time.time() - t1
    print(f"  Gen done: {gen_time:.1f}s")

    pipe.save_story_images(images, args.story, board.panels, str(output_dir))

    # Evaluation
    print("\n[Phase 3] Evaluation...")
    try:
        from storygen.evaluation_hub.metric_clip import CLIPEvaluator
        from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator

        clip_eval = CLIPEvaluator(device=f"cuda:{args.gpu}")
        prompts = [p.enhanced_prompt or p.raw_prompt for p in board.panels]
        clip_scores = clip_eval.compute_similarity(images, prompts)
        avg_clip = sum(clip_scores) / len(clip_scores)

        if len(images) > 1:
            cons_eval = ConsistencyEvaluator(device=f"cuda:{args.gpu}", metric="lpips")
            cons_scores = []
            for i in range(len(images) - 1):
                dist = cons_eval.compute_lpips_similarity(images[i], images[i + 1])
                cons_scores.append(1 - float(dist))
            avg_cons = sum(cons_scores) / len(cons_scores)
        else:
            avg_cons = 1.0
        overall = 0.6 * avg_clip + 0.4 * avg_cons
        print(f"  CLIP={avg_clip:.4f}  Consist={avg_cons:.4f}  Overall={overall:.4f}")
    except Exception as e:
        print(f"  Eval error: {e}")
        avg_clip, avg_cons, overall = 0, 0, 0

    result = {
        "story": args.story, "sca": cs, "window": args.window,
        "clip": avg_clip, "consist": avg_cons, "overall": overall,
        "parse_time": parse_time, "gen_time": gen_time,
    }
    report_path = output_dir / f"{args.story}_report.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"\nReport: {report_path}")
    print(f"Summary: story={args.story} CLIP={avg_clip:.4f} Consist={avg_cons:.4f} Overall={overall:.4f}")

if __name__ == "__main__":
    main()
