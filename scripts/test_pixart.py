"""
Test PixArt-Σ pipeline on target stories (04, 05, 07, 14).
Compare with SDXL baseline for clothing consistency.

Usage:
    python scripts/test_pixart.py [--story 04] [--all]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

from storygen.script_director.llm_parser import LLMScriptParser
from storygen.script_director.prompt_enhancer import PromptEnhancer
from storygen.core_generator.pipeline_pixart import PixArtGenerationPipeline

TARGET_STORIES = ["04", "05", "07", "14"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", type=str, default="04", help="Single story ID")
    parser.add_argument("--all", action="store_true", help="Run all target stories")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--consistency", type=float, default=0.14,
                        help="SCA consistency strength (0=disabled)")
    parser.add_argument("--window", type=int, default=1, help="SCA window size")
    parser.add_argument("--output_dir", type=str, default="outputs/pixart_test")
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and compose prompts without generation")
    return parser.parse_args()

def main():
    args = parse_args()

    stories = TARGET_STORIES if args.all else [args.story]
    print(f"Target stories: {stories}")

    script_dir = Path("storygen/data/TaskA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_parser = LLMScriptParser(llm_backend="local", model_name="llama3:70b")
    enhancer = PromptEnhancer()
    results = []

    for story_id in stories:
        script_path = script_dir / f"{story_id}.txt"
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing story {story_id}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            board = llm_parser.process_script_file(str(script_path))
            enhanced_prompts = enhancer.process_entire_story(board)
            parse_time = time.time() - t0
            print(f"Parse time: {parse_time:.1f}s, {len(board.panels)} panels")

            if args.dry_run:
                # Compose prompts using PixArt pipeline's compiler
                config = {
                    "consistency_strength": args.consistency,
                    "sca_window_size": args.window,
                    "sca_start_ratio": 0.25,
                    "generation_params": {
                        "num_steps": args.steps,
                        "guidance_scale": args.guidance,
                    },
                    "height": 1024,
                    "width": 1024,
                }
                pipe = PixArtGenerationPipeline(config)
                story_state = getattr(board, "story_state", {}) or {}

                print(f"\nCompiled prompts:")
                for i, panel in enumerate(board.panels):
                    prompt = pipe._compose_prompt(
                        panel=panel,
                        global_style=board.global_style,
                        characters=board.characters,
                        panel_index=i,
                        all_panels=board.panels,
                        consistency_constraints=board.consistency_constraints,
                        story_state=story_state,
                        return_plan=False,
                    )
                    print(f"  Panel {i+1}: {prompt[:200]}...")
                continue

            config = {
                "consistency_strength": args.consistency,
                "sca_window_size": args.window,
                "sca_start_ratio": 0.25,
                "generation_params": {
                    "num_steps": args.steps,
                    "guidance_scale": args.guidance,
                },
                "height": 1024,
                "width": 1024,
            }
            pipe = PixArtGenerationPipeline(config)

            gen_t0 = time.time()
            images, _ = pipe.generate_story(board, seed=42)
            gen_time = time.time() - gen_t0

            story_output_dir = output_dir / f"story_{story_id}"
            pipe.save_story_images(images, story_id, board.panels, str(output_dir))

            # Evaluate if CLIP evaluator is available
            try:
                from storygen.evaluation_hub.metric_clip import CLIPEvaluator
                from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator

                clip_eval = CLIPEvaluator(device='cuda:0')
                prompts = []
                for i, panel in enumerate(board.panels):
                    prompt = panel.enhanced_prompt or panel.raw_prompt
                    prompts.append(prompt)

                clip_scores = clip_eval.compute_similarity(images, prompts)
                avg_clip = sum(clip_scores) / len(clip_scores)

                if len(images) > 1:
                    cons_eval = ConsistencyEvaluator(device='cuda:0', metric='lpips')
                    cons_scores = []
                    for i in range(len(images) - 1):
                        dist = cons_eval.compute_lpips_similarity(images[i], images[i + 1])
                        cons_scores.append(1 - float(dist))
                    avg_cons = sum(cons_scores) / len(cons_scores)
                else:
                    avg_cons = 1.0

                overall = 0.6 * avg_clip + 0.4 * avg_cons
                print(f"\n[Eval] Story {story_id}:")
                print(f"  CLIP={avg_clip:.4f}  Consist={avg_cons:.4f}  Overall={overall:.4f}")
            except Exception as e:
                print(f"Evaluation error: {e}")
                avg_clip, avg_cons, overall = 0, 0, 0

            results.append({
                "story_id": story_id,
                "parse_time": parse_time,
                "gen_time": gen_time,
                "avg_clip_score": avg_clip,
                "avg_consistency": avg_cons,
                "overall_score": overall,
                "num_panels": len(board.panels),
            })

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {story_id}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        report_path = output_dir / "pixart_report.json"
        report_path.write_text(json.dumps(results, indent=2))
        print(f"\nReport saved: {report_path}")
        print("\nSummary:")
        for r in results:
            print(f"  {r['story_id']}: CLIP={r['avg_clip_score']:.4f}, "
                  f"Consist={r['avg_consistency']:.4f}, Overall={r['overall_score']:.4f}, "
                  f"Time={r['gen_time']:.1f}s")


if __name__ == "__main__":
    main()
