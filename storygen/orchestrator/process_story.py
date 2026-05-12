"""
Shared process_story function — single-story pipeline.
Used by both sequential_batch.py and parallel_batch_runner.py.
"""

import json, os, time, gc, torch, subprocess, glob
from pathlib import Path
from PIL import Image
import numpy as np

from storygen.core_generator.config_defaults import build_generation_config

GENERIC_PANEL_ACTIONS = {
    "focused expression",
    "relaxed posture",
    "walks into a new place",
    "continues its task",
    "works in a factory",
    "looking at nina",
    "looks around",
    "interacts with staff",
}

FIXTURE_CARRY_MARKERS = {
    "information desk",
    "counter",
    "counters",
    "signage",
    "kiosk",
    "machine",
    "machines",
    "conveyor",
    "screen",
    "screens",
    "monitor",
    "monitors",
}

NONHUMAN_SCRIPT_MARKERS = {
    "bird",
    "robot",
    "cat",
    "dog",
    "animal",
    "puppy",
    "kitten",
    "horse",
    "bear",
    "fox",
    "rabbit",
    "creature",
}

AIRPORT_SCRIPT_MARKERS = {
    "airport",
    "terminal",
    "gate",
    "flight",
    "luggage",
    "suitcase",
    "carousel",
}

BUS_SCRIPT_MARKERS = {
    "bus",
    "bus stop",
}

EXPLICIT_TRANSIT_ANCHORS = {
    "ticket",
    "gate",
    "information desk",
}


def _panel_states(board) -> list:
    return ((getattr(board, "story_state", {}) or {}).get("panel_states") or [])


def _character_dict(board) -> dict:
    return getattr(board, "characters", {}) or {}


def _count_generic_actions(board) -> int:
    count = 0
    for panel_state in _panel_states(board):
        actions = panel_state.get("action_beats") or []
        if actions and str(actions[0]).strip().lower() in GENERIC_PANEL_ACTIONS:
            count += 1
    return count


def _has_bad_scene_carry(board) -> bool:
    states = _panel_states(board)
    for idx in range(1, len(states)):
        prev_segment = str(states[idx - 1].get("scene_segment", ""))
        curr_segment = str(states[idx].get("scene_segment", ""))
        if prev_segment == curr_segment:
            continue
        continuity_text = " ".join(states[idx].get("continuity_from_prev") or []).lower()
        if "carry over" not in continuity_text:
            continue
        if any(marker in continuity_text for marker in FIXTURE_CARRY_MARKERS):
            return True
    return False


def _story_entity_types(board) -> set:
    entity_types = set()
    for char in _character_dict(board).values():
        entity_type = str(getattr(char, "entity_type", "") or "").lower()
        if entity_type:
            entity_types.add(entity_type)
    return entity_types


def _is_multi_character_story(board) -> bool:
    return any((panel_state.get("expected_count") or 0) > 1 for panel_state in _panel_states(board))


def _is_nonhuman_story(script_text: str, board) -> bool:
    if any(marker in script_text for marker in NONHUMAN_SCRIPT_MARKERS):
        return True
    return any(entity_type not in {"", "human"} for entity_type in _story_entity_types(board))


def _is_single_nonhuman_story(script_text: str, board) -> bool:
    return _is_nonhuman_story(script_text, board) and not _is_multi_character_story(board)


def _is_multi_human_story(board) -> bool:
    entity_types = _story_entity_types(board)
    return _is_multi_character_story(board) and entity_types and entity_types.issubset({"human"})


def _is_generic_airport_story(script_text: str, board) -> bool:
    story_text = " ".join(
        [
            script_text,
            " ".join(str(getattr(panel, "setting", "") or "") for panel in (getattr(board, "panels", []) or [])),
            " ".join((getattr(board, "story_state", {}) or {}).get("global_location_graph", []) or []),
        ]
    ).lower()
    return any(marker in story_text for marker in AIRPORT_SCRIPT_MARKERS) and not any(
        marker in script_text for marker in EXPLICIT_TRANSIT_ANCHORS
    )


def _is_bus_story(script_text: str, board) -> bool:
    story_text = " ".join(
        [
            script_text,
            " ".join(str(getattr(panel, "setting", "") or "") for panel in (getattr(board, "panels", []) or [])),
            " ".join((getattr(board, "story_state", {}) or {}).get("global_location_graph", []) or []),
        ]
    ).lower()
    return any(marker in story_text for marker in BUS_SCRIPT_MARKERS)


def _tag_parser_variant(board, variant: str):
    story_state = getattr(board, "story_state", None)
    if isinstance(story_state, dict):
        story_state["parser_variant"] = variant
    return board

def get_cache_dir() -> str:
    from storygen.utils.mirror_config import setup_china_mirrors, get_models_cache_dir
    setup_china_mirrors()
    return str(get_models_cache_dir())

def parse_story(script_file: str, gpu_id: int = 0):
    """Parse a single story script using Qwen on specified GPU.
    Returns (board, board_dict) where board is the ProductionBoard object."""
    from storygen.script_director.llm_parser_local import create_qwen_parser
    script_text = Path(script_file).read_text(encoding="utf-8").lower()
    parser = create_qwen_parser(device_map=f'cuda:{gpu_id}')
    with parser:
        greedy_board = parser.process_script_file(script_file)

        should_try_sampled = (
            _is_single_nonhuman_story(script_text, greedy_board)
            or _is_multi_human_story(greedy_board)
            or _is_bus_story(script_text, greedy_board)
            or _is_generic_airport_story(script_text, greedy_board)
        )
        if should_try_sampled:
            original_settings = (
                parser.do_sample,
                parser.temperature,
                parser.top_p,
                parser.top_k,
                parser.seed_from_input,
            )
            parser.do_sample = True
            parser.temperature = 0.7
            parser.top_p = 0.9
            parser.top_k = 50
            parser.seed_from_input = True
            sampled_board = parser.process_script_file(script_file)
            parser.do_sample, parser.temperature, parser.top_p, parser.top_k, parser.seed_from_input = original_settings

            choose_sampled = False
            if not _has_bad_scene_carry(sampled_board):
                greedy_generic = _count_generic_actions(greedy_board)
                sampled_generic = _count_generic_actions(sampled_board)
                if sampled_generic + 1 < greedy_generic:
                    choose_sampled = True
                elif _is_single_nonhuman_story(script_text, sampled_board) or _is_multi_human_story(sampled_board):
                    choose_sampled = True
                elif _is_bus_story(script_text, sampled_board):
                    choose_sampled = True
                elif _is_generic_airport_story(script_text, sampled_board):
                    choose_sampled = True

            board = _tag_parser_variant(sampled_board if choose_sampled else greedy_board, "sampled" if choose_sampled else "greedy")
        else:
            board = _tag_parser_variant(greedy_board, "greedy")
    del parser
    torch.cuda.empty_cache()
    gc.collect()
    return board

def generate_story(board, config: dict, device: str = "cuda:0"):
    """Generate story images using SDXL on specified device.
    Returns list of PIL Images."""
    from storygen.core_generator.pipeline import NarrativeGenerationPipeline
    pipe = NarrativeGenerationPipeline(config)
    images, _ = pipe.generate_story(board, seed=42)
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return images

def evaluate_story(images, board, device: str = "cuda:0"):
    """Evaluate generated images with CLIP + LPIPS.
    Returns dict with avg_clip_score, avg_consistency, per-frame scores."""
    from storygen.evaluation_hub.metric_clip import CLIPEvaluator
    from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator
    from storygen.evaluation_hub.state_eval import (
        derive_failure_tags,
        evaluate_prompt_state_alignment,
    )
    
    clip = CLIPEvaluator(device=device)
    clip_scores = []
    prompts = []
    for i, panel in enumerate(board.panels):
        if getattr(board, "render_plan", None) and i < len(board.render_plan):
            prompt = board.render_plan[i].get("compiled_prompt") or panel.enhanced_prompt or panel.raw_prompt
        else:
            prompt = panel.enhanced_prompt or panel.raw_prompt
        prompts.append(prompt)
        score = clip.compute_similarity([images[i]], [prompt])[0]
        clip_scores.append(float(score))
    avg_clip = sum(clip_scores) / len(clip_scores)
    del clip
    torch.cuda.empty_cache()
    
    if len(images) > 1:
        cons = ConsistencyEvaluator(device=device, metric="lpips")
        lpips_scores = []
        for i in range(len(images) - 1):
            dist = cons.compute_lpips_similarity(images[i], images[i + 1])
            lpips_scores.append(1 - float(dist))
        avg_cons = sum(lpips_scores) / len(lpips_scores)
        del cons
        torch.cuda.empty_cache()
    else:
        avg_cons = 1.0
    story_state = getattr(board, "story_state", {}) or {}
    state_alignment = evaluate_prompt_state_alignment(story_state, prompts) if story_state else {
        "score": 0.0,
        "checks": [],
        "issues": [],
        "panel_reports": [],
        "panel_summary": [],
        "entity_summary": {},
    }
    failure_tags = derive_failure_tags(state_alignment, vision_summary="")

    return {
        "prompts": prompts,
        "clip_scores": clip_scores,
        "avg_clip_score": avg_clip,
        "avg_consistency": avg_cons,
        "overall_score": 0.6 * avg_clip + 0.4 * avg_cons,
        "state_alignment_score": state_alignment.get("score", 0.0),
        "state_alignment_issues": state_alignment.get("issues", []),
        "panel_alignment": state_alignment.get("panel_reports", []),
        "panel_alignment_summary": state_alignment.get("panel_summary", []),
        "entity_summary": state_alignment.get("entity_summary", {}),
        "failure_tags": failure_tags,
        "parser_variant": story_state.get("parser_variant"),
    }

def save_results(images, metrics, board, output_dir: Path, gen_time: float, parse_time: float):
    """Save generated images, evaluation, and storyboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from storygen.utils.image_utils import remove_white_borders, create_storyboard
    
    for i, img in enumerate(images, 1):
        clean = remove_white_borders(img)
        clean.save(output_dir / f"frame_{i:02d}.png")

    production_board_data = {
        "story_id": board.story_id,
        "global_style": board.global_style,
        "characters": {name: char.__dict__ for name, char in board.characters.items()},
        "panels": [panel.__dict__ for panel in board.panels],
        "consistency_constraints": board.consistency_constraints,
        "narrative_arc": board.narrative_arc,
        "story_state": getattr(board, "story_state", {}),
        "render_plan": getattr(board, "render_plan", []),
    }
    with open(output_dir / "production_board.json", "w") as f:
        json.dump(production_board_data, f, indent=2)
    
    # Save evaluation
    eval_data = {
        "script": str(metrics.get("script", "")),
        "num_panels": len(board.panels),
        "num_characters": len(board.characters),
        "clip_scores": metrics["clip_scores"],
        "prompts": metrics.get("prompts", []),
        "avg_clip_score": metrics["avg_clip_score"],
        "avg_consistency": metrics["avg_consistency"],
        "overall_score": metrics["overall_score"],
        "state_alignment_score": metrics.get("state_alignment_score", 0.0),
        "state_alignment_issues": metrics.get("state_alignment_issues", []),
        "panel_alignment_summary": metrics.get("panel_alignment_summary", []),
        "entity_summary": metrics.get("entity_summary", {}),
        "failure_tags": metrics.get("failure_tags", []),
        "parser_variant": metrics.get("parser_variant"),
        "parse_time_s": round(parse_time, 1),
        "generation_time_s": round(gen_time, 1),
    }
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(eval_data, f, indent=2)

    with open(output_dir / "story_state.json", "w") as f:
        json.dump(getattr(board, "story_state", {}), f, indent=2)

    with open(output_dir / "render_plan.json", "w") as f:
        json.dump(getattr(board, "render_plan", []), f, indent=2)

    with open(output_dir / "panel_alignment.json", "w") as f:
        json.dump(metrics.get("panel_alignment", []), f, indent=2)
    
    # Create storyboard
    try:
        labels = [f"Scene {p.panel_id}: {p.shot_type}" for p in board.panels]
        storyboard = create_storyboard(images, labels, image_size=(512, 512))
        storyboard.save(output_dir / "storyboard.png")
    except Exception as e:
        print(f"  [Save] Storyboard error: {e}", flush=True)

def process_story(script_file: str, output_base: Path = None, gpu_id: int = 0,
                  config: dict = None) -> dict:
    """
    Full single-story pipeline: parse → generate → evaluate → save.
    Uses specified GPU for all operations.
    
    Args:
        script_file: Path to the story script
        output_base: Base directory for outputs (default: outputs/taskA_batch)
        gpu_id: GPU to use (mapped via CUDA_VISIBLE_DEVICES)
        config: Generation configuration (uses defaults if None)
    
    Returns:
        dict with story metrics
    """
    if output_base is None:
        output_base = Path("outputs/taskA_batch")
    if config is None:
        config = build_generation_config(device=f"cuda:{gpu_id}")
    else:
        config = build_generation_config(config, device=f"cuda:{gpu_id}")
    
    script_path = Path(script_file)
    sid = script_path.stem
    output_dir = output_base / sid
    
    print(f"\n[{gpu_id}] === {sid} ===", flush=True)
    
    # Phase 1: Parse
    t0 = time.time()
    try:
        board = parse_story(script_file, gpu_id)
    except Exception as e:
        print(f"[{gpu_id}] {sid}: PARSE FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        return {"script": str(script_file), "error": str(e), "overall_score": 0.0}
    parse_time = time.time() - t0
    print(f"  [{gpu_id}] Parse: {parse_time:.1f}s", flush=True)
    
    # Phase 2: Generate
    gen_t0 = time.time()
    try:
        images = generate_story(board, config, f"cuda:{gpu_id}")
    except Exception as e:
        print(f"[{gpu_id}] {sid}: GENERATION FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        return {"script": str(script_file), "error": str(e), "overall_score": 0.0}
    gen_time = time.time() - gen_t0
    print(f"  [{gpu_id}] Gen: {gen_time:.1f}s", flush=True)
    
    # Phase 3: Evaluate
    try:
        metrics = evaluate_story(images, board, f"cuda:{gpu_id}")
    except Exception as e:
        print(f"[{gpu_id}] {sid}: EVAL FAILED: {e}", flush=True)
        # Still save images even if eval fails
        metrics = {"clip_scores": [0.0]*3, "avg_clip_score": 0.0,
                   "avg_consistency": 0.0, "overall_score": 0.0}
    
    metrics["script"] = str(script_file)
    metrics["generation_time_s"] = round(gen_time, 1)
    metrics["parse_time_s"] = round(parse_time, 1)
    
    # Phase 4: Save
    save_results(images, metrics, board, output_dir, gen_time, parse_time)
    
    print(f"  [{gpu_id}] {sid}: CLIP={metrics['avg_clip_score']:.3f} "
          f"Consist={metrics['avg_consistency']:.3f} "
          f"Overall={metrics['overall_score']:.3f}", flush=True)
    
    return metrics


def process_story_subprocess(script_file: str, gpu_id: int, output_base: str) -> dict:
    """Run process_story in a subprocess with isolated GPU.
    Returns parsed metrics dict from stdout."""
    import subprocess, json
    cmd = [
        "python3", "-c", f"""
import sys, json
sys.path.insert(0, '.')
from storygen.orchestrator.process_story import process_story
result = process_story('{script_file}', Path('{output_base}'), gpu_id={gpu_id})
print("STORY_DONE:" + json.dumps(result))
"""]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    for line in result.stdout.strip().split("\n"):
        if line.startswith("STORY_DONE:"):
            return json.loads(line[11:])
    print(f"Subprocess stderr: {result.stderr[:500]}", flush=True)
    return {"script": script_file, "error": "subprocess_failed", "overall_score": 0.0}
