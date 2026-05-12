#!/usr/bin/env python3
"""
Auto Iterate — Autonomous experimentation engine.
Orchestrates: generate → eval → VLM diag → meta-plan → iterate.
GPU-aware: waits for available GPUs, manages locks.
Memory-aware: logs all results for self-improvement.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/../ -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpu_scheduler import get_available_gpu, acquire_gpu, wait_for_gpu, release_gpu, release_all_gpus, print_status


# ── Memory via CLI (no MCP dependency) ──────────────────────────────

MEMORY_DB = str(PROJECT_ROOT / ".storygen_memory.db")
PROJECT_STATE_PATH = PROJECT_ROOT / ".opencode" / "project_state.md"
FAILURE_ATLAS_PATH = PROJECT_ROOT / "outputs" / "taskA_batch" / "failure_atlas.json"


def memory_add(key: str, value: str, category: str = "experiment", tags: list = None):
    """Add memory via direct SQLite (MCP-independent)."""
    import sqlite3, json
    tags_json = json.dumps(tags or [])
    conn = sqlite3.connect(MEMORY_DB)
    try:
        conn.execute(
            "INSERT INTO memories (key, value, category, tags, user_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'auto_iterate', ?, ?)",
            (key, value, category, tags_json, time.time(), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def memory_search(query: str, category: str = None, limit: int = 5) -> list:
    """Search memories via SQLite FTS5."""
    import sqlite3, json
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    try:
        if category:
            rows = conn.execute(
                "SELECT m.key, m.value, m.category, m.tags FROM memories_fts f "
                "JOIN memories m ON f.rowid = m.id "
                "WHERE memories_fts MATCH ? AND m.category = ? "
                "ORDER BY rank LIMIT ?",
                (query, category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT m.key, m.value, m.category, m.tags FROM memories_fts f "
                "JOIN memories m ON f.rowid = m.id "
                "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        results = []
        for r in rows:
            results.append({"key": r["key"], "value": r["value"],
                            "category": r["category"], "tags": json.loads(r["tags"])})
        return results
    finally:
        conn.close()


# ── Phase Configurations ──────────────────────────────────────────

PHASES = {
    "A2.0": {
        "name": "Face Embedding",
        "agent": "face-lock",
        "stories": ["11", "06"],  # Single char + multi char
        "priority": "high",
    },
    "A3.0": {
        "name": "Non-human Fix",
        "agent": "lora-train",
        "stories": ["13", "extra_06"],
        "priority": "medium",
    },
}


def load_current_baseline() -> dict:
    """Read the latest trusted quantitative row from project_state.md."""
    fallback = {"clip": 0.313449, "consistency": 0.393319, "overall": 0.345397}
    if not PROJECT_STATE_PATH.exists():
        return fallback
    text = PROJECT_STATE_PATH.read_text(encoding="utf-8")
    rows = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("| 20") and line.count("|") >= 7
    ]
    if not rows:
        return fallback
    latest = rows[-1]
    columns = [part.strip() for part in latest.strip("|").split("|")]
    if len(columns) < 5:
        return fallback
    try:
        return {
            "clip": float(columns[1]),
            "consistency": float(columns[2]),
            "overall": float(columns[3]),
        }
    except ValueError:
        return fallback


def load_failure_atlas_targets(limit: int = 5) -> list[str]:
    """Pick the current lowest project-rubric stories from the saved batch atlas."""
    if not FAILURE_ATLAS_PATH.exists():
        return []
    try:
        atlas = json.loads(FAILURE_ATLAS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    lowest = atlas.get("lowest_project_stories", []) or atlas.get("lowest_stories", [])
    targets = []
    for item in lowest:
        story_id = str(item.get("story_id", "")).strip()
        if story_id and story_id not in targets:
            targets.append(story_id)
        if len(targets) >= limit:
            break
    return targets


# ── Subprocess Helpers ────────────────────────────────────────────

def run_with_gpu(cmd: list, min_mem_mb: int, holder: str, poll: int = 30) -> tuple:
    """Run a command on an available GPU. Wait if needed."""
    gpu = wait_for_gpu(min_mem_mb, poll_interval=poll, holder=holder)
    # Do NOT set CUDA_VISIBLE_DEVICES - let the subprocess select GPU via scheduler
    env = os.environ.copy()
    # Remove CUDA_VISIBLE_DEVICES if already set to avoid conflict with internal scheduler
    if "CUDA_VISIBLE_DEVICES" in env:
        del env["CUDA_VISIBLE_DEVICES"]
    print(f"[AutoIterate] ▶ Running on GPU {gpu}: {' '.join(cmd[:3])}...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    release_gpu(gpu)
    if proc.returncode != 0:
        print(f"[AutoIterate] ⚠️  Exit code {proc.returncode}")
        if proc.stderr:
            for line in proc.stderr.strip().split("\n")[-10:]:
                print(f"  STDERR: {line}")
        if proc.stdout:
            for line in proc.stdout.strip().split("\n")[-5:]:
                print(f"  STDOUT: {line}")
    return proc.returncode, proc.stdout, proc.stderr


# ── Stage Runners ─────────────────────────────────────────────────

def run_generation(story_id: str, wait: bool = True) -> bool:
    """Generate story images using run_taska_batch (handles its own GPU scheduling)."""
    print(f"\n{'='*60}")
    print(f"[Generate] story {story_id}")
    print(f"{'='*60}")

    cmd = [
        "python3", str(PROJECT_ROOT / "run_taska_batch.py"),
        "--story", story_id,
    ]
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in env:
        del env["CUDA_VISIBLE_DEVICES"]
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print(f"[AutoIterate] Generating story {story_id}...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    print(f"[AutoIterate] stdout: {proc.stdout[-300:]}" if proc.stdout else "")
    if proc.stderr:
        print(f"[AutoIterate] stderr: {proc.stderr[-300:]}")

    success = proc.returncode == 0
    memory_add(f"gen_{story_id}", f"success={success}",
               category="experiment", tags=["generation", story_id])
    print(f"[Generate] {'✅' if success else '❌'} (exit={proc.returncode})")
    return success


def run_evaluation(story_id: str) -> dict:
    """Run CLIP + LPIPS evaluation. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"[Eval] story {story_id}")
    print(f"{'='*60}")

    gpu = wait_for_gpu(2048, holder=f"eval_{story_id}")
    cmd = [
        "python3", "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator
from storygen.evaluation_hub.state_eval import evaluate_prompt_state_alignment
from PIL import Image
from pathlib import Path

base = Path('{PROJECT_ROOT}/storygen/outputs/taskA_batch/{story_id}')
images = [Image.open(p) for p in sorted(base.glob('frame_*.png'))]
if not images:
    print(json.dumps({{"error": "No frames found"}}))
    exit(1)

prompts = []
eval_file = base / "evaluation.json"
if eval_file.exists():
    with open(eval_file) as f:
        eval_data = json.load(f)
    prompts = eval_data.get("prompts", [])

render_plan_file = base / "render_plan.json"
if render_plan_file.exists():
    with open(render_plan_file) as f:
        render_plan = json.load(f)
    if render_plan:
        prompts = [item.get("compiled_prompt", "") for item in render_plan]

if not prompts:
    board_file = base / "production_board.json"
    if board_file.exists():
        with open(board_file) as f:
            board = json.load(f)
        prompts = [
            panel.get("enhanced_prompt") or panel.get("raw_prompt") or f"Frame {{i+1}}"
            for i, panel in enumerate(board.get("panels", []))
        ]

if len(prompts) < len(images):
    prompts.extend([f"Frame {{i+1}}" for i in range(len(prompts), len(images))])

story_state = {{}}
story_state_file = base / "story_state.json"
if story_state_file.exists():
    with open(story_state_file) as f:
        story_state = json.load(f)
state_alignment = evaluate_prompt_state_alignment(story_state, prompts) if story_state else {{"score": 0.0, "issues": []}}

clip = CLIPEvaluator(device='cuda')
scores = [clip.compute_similarity([img], [prompt])[0] for img, prompt in zip(images, prompts)]
avg_clip = sum(scores)/len(scores)

cons = ConsistencyEvaluator(device='cuda', metric='lpips')
lpips_scores = []
for i in range(len(images)-1):
    d = cons.compute_lpips_similarity(images[i], images[i+1])
    lpips_scores.append(1 - d)
avg_cons = sum(lpips_scores)/len(lpips_scores) if lpips_scores else 1.0

print(json.dumps({{
    "story_id": "{story_id}",
    "prompts": prompts,
    "avg_clip_score": round(avg_clip, 4),
    "avg_consistency": round(avg_cons, 4),
    "overall": round(0.6*avg_clip + 0.4*avg_cons, 4),
    "state_alignment_score": round(state_alignment.get("score", 0.0), 4),
    "state_alignment_issues": state_alignment.get("issues", [])[:10],
}}))
""",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    release_gpu(gpu)

    try:
        metrics = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        metrics = {"error": proc.stdout[:200], "stderr": proc.stderr[:200]}

    memory_add(f"eval_{story_id}", json.dumps(metrics),
               category="experiment", tags=["evaluation", story_id])
    print(f"[Eval] {'✅' if 'avg_clip_score' in metrics else '❌'} "
          f"CLIP={metrics.get('avg_clip_score','?'):.3f}" if 'avg_clip_score' in metrics else "[Eval] ❌")
    return metrics


def run_vision_eval(story_id: str, version_tag: str = "auto_v1") -> dict:
    """Run VLM diagnostic (fallback if VLM unavailable)."""
    print(f"\n[VisionEval] story {story_id}")
    gpu = acquire_gpu(4096, holder=f"vlm_{story_id}")
    try:
        cmd = [
            "python3", "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from storygen.evaluation_hub.vision_eval import evaluate_all_stories
result = evaluate_all_stories(
    base_dir='{PROJECT_ROOT}/storygen/outputs/taskA_batch',
    version_tag='{version_tag}',
    story_ids=['{story_id}'],
)
print(result)
""",
        ]
        if gpu is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            release_gpu(gpu)
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        return {"output": proc.stdout[:500], "exit": proc.returncode}
    except Exception as e:
        return {"error": str(e)}


# ── Meta-Planner (via rules, no API call needed) ──────────────────

def meta_plan(story_id: str, metrics: dict, baseline: dict) -> dict:
    """Rule-based meta-planner. Compares against baseline, decides next action."""
    clip = metrics.get("avg_clip_score", 0)
    consist = metrics.get("avg_consistency", 0)
    overall = metrics.get("overall", 0)

    base_clip = baseline.get("clip", 0.273)
    base_consist = baseline.get("consistency", 0.424)
    base_overall = baseline.get("overall", 0.333)

    verdicts = []
    if overall >= base_overall:
        verdicts.append(("proceed", f"Overall {overall:.3f} >= baseline {base_overall:.3f}"))
    else:
        verdicts.append(("investigate", f"Overall {overall:.3f} < baseline {base_overall:.3f}"))

    if clip < base_clip - 0.02:
        verdicts.append(("investigate", f"CLIP {clip:.3f} dropped >0.02 from {base_clip:.3f}"))
    if consist < base_consist - 0.02:
        verdicts.append(("investigate", f"Consistency {consist:.3f} dropped >0.02 from {base_consist:.3f}"))
    if metrics.get("state_alignment_score", 1.0) < 0.7:
        verdicts.append((
            "investigate",
            f"State alignment {metrics.get('state_alignment_score', 0.0):.3f} < 0.700",
        ))

    # Final verdict
    if all(v[0] == "proceed" for v in verdicts):
        final = "proceed"
    elif any(v[0] == "investigate" for v in verdicts):
        final = "investigate"
    else:
        final = "rollback"

    return {
        "story_id": story_id,
        "verdict": final,
        "reasons": [v[1] for v in verdicts],
        "metrics": metrics,
        "baseline": baseline,
    }

# ── Main Iteration Loop ──────────────────────────────────────────

BASELINE = load_current_baseline()


def auto_iterate(
    phase: str = "A2.0",
    max_iterations: int = 5,
    story_ids: list = None,
    wait: bool = True,
):
    """Main autonomous iteration loop."""
    phase_info = PHASES.get(phase, {"name": phase, "agent": "unknown"})
    atlas_targets = [] if story_ids else load_failure_atlas_targets()
    stories = story_ids or atlas_targets or phase_info.get("stories", [])
    agent = phase_info.get("agent", "unknown")

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  Auto Iterate — {phase}: {phase_info['name']}                  ║
║  Stories: {stories}                                          ║
║  Max iters: {max_iterations}                                     ║
║  Baseline: {BASELINE['overall']:.3f} overall                         ║
╚═══════════════════════════════════════════════════════════╝
    """)

    memory_add(f"iter_{phase}_start", json.dumps({
        "phase": phase, "stories": stories, "agent": agent,
        "baseline": BASELINE,
        "atlas_targets": atlas_targets,
        "timestamp": datetime.now().isoformat(),
    }), category="experiment")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration}/{max_iterations}")
        print(f"{'#'*60}")
        memory_add(f"iter_{phase}_{iteration}", f"started",
                   category="experiment", tags=[phase, f"iter_{iteration}"])

        # Check GPU availability
        print("\n[GPU Status]")
        gpus = get_available_gpu(20000)
        print(f"  GPU with 20GB+ free: {'yes' if gpus is not None else 'no'}")

        for sid in stories:
            print(f"\n── Story {sid} ──")

            # 1. Generate
            success = run_generation(sid, wait=wait)
            if not success:
                if not wait:
                    print(f"[Skip] No GPU for {sid}, will retry later")
                    continue
                else:
                    print(f"[Fail] Generation failed for {sid}")
                    continue

            # 2. Evaluate
            metrics = run_evaluation(sid)
            if "error" in metrics:
                print(f"[Skip] Eval failed for {sid}: {metrics['error']}")
                continue

            # 3. VLM diagnostic
            vlm_result = run_vision_eval(sid)

            # 4. Meta-plan
            plan = meta_plan(sid, metrics, BASELINE)
            print(f"\n[Plan] {plan['verdict'].upper()}: {plan['reasons']}")

            memory_add(f"iter_{phase}_{iteration}_{sid}", json.dumps({
                "iteration": iteration,
                "story_id": sid,
                "metrics": metrics,
                "verdict": plan["verdict"],
                "reasons": plan["reasons"],
                "vlm": {"available": "error" not in vlm_result},
            }), category="experiment", tags=[phase, sid, plan["verdict"]])

            if plan["verdict"] == "investigate":
                print("[Action] Metrics below baseline — generating investigation report")
                # In a full auto system: call @meta-planner via agent
                # For now: log the detailed comparison
                print(f"  CLIP: {metrics.get('avg_clip_score', '?'):.3f} vs baseline {BASELINE['clip']:.3f}")
                print(f"  Consistency: {metrics.get('avg_consistency', '?'):.3f} vs baseline {BASELINE['consistency']:.3f}")

        # Phase complete
        print(f"\n✅ Iteration {iteration}/{max_iterations} complete")

    print(f"\n{'='*60}")
    print(f"Phase {phase} complete after {max_iterations} iterations")
    print(f"{'='*60}")

    memory_add(f"iter_{phase}_done", json.dumps({
        "phase": phase, "iterations": max_iterations,
        "stories": stories, "timestamp": datetime.now().isoformat(),
    }), category="experiment")

    release_all_gpus()
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto Iterate")
    parser.add_argument("--phase", default="A2.0", choices=list(PHASES.keys()) + ["custom"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--story", nargs="+", help="Story IDs to process")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for GPU")
    args = parser.parse_args()

    auto_iterate(
        phase=args.phase,
        max_iterations=args.iterations,
        story_ids=args.story,
        wait=not args.no_wait,
    )
