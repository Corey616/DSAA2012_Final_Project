#!/usr/bin/env python3
"""
Iteration Watchdog — Monitors model downloads + GPU availability,
then triggers auto-iteration. Keeps looping until SOTA is achieved.
SOTA target: CLIP >= 0.30, Consistency >= 0.50, Pass Rate >= 90%
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/data/lizhenzhuo/programs/DSAA2012/DSAA2012_Final_Project")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

SOTA = {
    "min_clip": 0.30,
    "min_consistency": 0.50,
    "min_overall": 0.38,
    "min_pass_rate": 0.90,
}


def _log(msg):
    print(f"[Watchdog] {msg}", flush=True)


def model_status():
    """Check if SDXL model is cached (local path or HF cache)."""
    local_path = Path("/data/lizhenzhuo/models/stable-diffusion-xl-base-1.0")
    if local_path.exists() and (local_path / "model_index.json").exists():
        unet_files = list((local_path / "unet").rglob("*.safetensors"))
        if unet_files and any(f.stat().st_size > 1e8 for f in unet_files):
            return True, f"SDXL ready at local path ({local_path})"

    cache = PROJECT / "storygen/models"
    sdxl_path = cache / "models--stabilityai--stable-diffusion-xl-base-1.0"
    if not sdxl_path.exists():
        return False, "SDXL cache not found"

    safetensors = list(sdxl_path.rglob("*.safetensors"))
    total_complete = sum(f.stat().st_size for f in safetensors if f.stat().st_size > 1e8)

    blobs = list(sdxl_path.glob("blobs/*"))
    total_blobs = sum(f.stat().st_size for f in blobs if f.is_file())
    incomplete_count = len([b for b in blobs if b.name.endswith('.incomplete')])

    ready = total_complete > 5e9
    msg = f"SDXL ready ({total_complete/1024**3:.1f}GB)" if ready else \
          f"SDXL partial (snap:{total_complete/1024**3:.1f}GB, blob:{total_blobs/1024**3:.1f}GB, {incomplete_count} pending)"
    return ready, msg


def gpu_available(min_mem_mb=8000):
    """Check if a GPU with enough memory exists."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[1]) >= min_mem_mb:
                return True, int(parts[0])
        return False, None
    except Exception as e:
        return False, str(e)


def run_auto_iterate():
    """Launch auto-iteration and wait for completion."""
    _log("Starting auto-iteration...")
    proc = subprocess.Popen(
        ["python3", "-u", str(PROJECT / "scripts/auto_iterate.py"),
         "--phase", "A2.0", "--iterations", "1", "--story", "11", "06"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    # Stream output
    for line in iter(proc.stdout.readline, b""):
        print(line.decode(errors="replace").rstrip(), flush=True)
    proc.wait()
    return proc.returncode


def check_against_sota(summary_path):
    """Compare latest eval against SOTA targets."""
    try:
        with open(summary_path) as f:
            data = json.load(f)
        results = [r for r in data.get("results", []) if "error" not in r]
        if not results:
            return False, "No valid results"

        avg_clip = sum(r["avg_clip_score"] for r in results) / len(results)
        avg_consist = sum(r["avg_consistency"] for r in results) / len(results)
        overalls = [r["overall_score"] for r in results]
        pass_rate = sum(1 for o in overalls if o >= 0.3) / len(overalls)

        status = {
            "clip": avg_clip, "consistency": avg_consist,
            "overall": sum(overalls) / len(overalls),
            "pass_rate": pass_rate,
        }

        gaps = []
        if avg_clip < SOTA["min_clip"]:
            gaps.append(f"CLIP {avg_clip:.3f} < {SOTA['min_clip']}")
        if avg_consist < SOTA["min_consistency"]:
            gaps.append(f"Consistency {avg_consist:.3f} < {SOTA['min_consistency']}")
        if status["overall"] < SOTA["min_overall"]:
            gaps.append(f"Overall {status['overall']:.3f} < {SOTA['min_overall']}")
        if pass_rate < SOTA["min_pass_rate"]:
            gaps.append(f"Pass rate {pass_rate:.1%} < {SOTA['min_pass_rate']:.0%}")

        if not gaps:
            return True, f"SOTA achieved! {status}"
        return False, f"SOTA not met: {'; '.join(gaps)}"
    except (FileNotFoundError, json.JSONDecodeError, ZeroDivisionError) as e:
        return False, f"Cannot evaluate: {e}"


def log_memory(msg):
    """Write to SQLite memory."""
    import sqlite3
    db = PROJECT / ".storygen_memory.db"
    try:
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO memories (key, value, category, tags, user_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"watchdog_{int(time.time())}", msg, "experiment",
             json.dumps(["watchdog"]), "watchdog", time.time(), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── Main Loop ──────────────────────────────────────────────

def main():
    _log("=" * 60)
    _log("Iteration Watchdog started")
    _log(f"SOTA targets: {SOTA}")
    _log("=" * 60)

    iteration = 0
    while True:
        iteration += 1
        _log(f"\n--- Check #{iteration} ---")

        # Step 1: Model readiness
        ready, status = model_status()
        _log(f"Model: {status}")

        if not ready:
            _log("Waiting for model download... (check every 60s)")
            log_memory(f"watchdog_wait_model:{status}")
            time.sleep(60)
            continue

        # Step 2: GPU availability
        gpu_ok, gpu_id = gpu_available()
        _log(f"GPU (8GB+): {gpu_ok} ({'GPU '+str(gpu_id) if gpu_ok else 'none'})")

        if not gpu_ok:
            _log("Waiting for GPU... (check every 60s)")
            log_memory(f"watchdog_wait_gpu:{status}")
            time.sleep(60)
            continue

        # Step 3: Check current results against SOTA
        summary_path = PROJECT / "storygen/outputs/taskA_batch/batch_summary.json"
        if summary_path.exists():
            sota_achieved, sota_msg = check_against_sota(summary_path)
            _log(f"SOTA check: {sota_msg}")
            log_memory(f"watchdog_sota:{sota_msg}")

            if sota_achieved:
                _log("\n" + "=" * 60)
                _log("🎉 SOTA ACHIEVED! All targets met.")
                _log("=" * 60)
                log_memory("watchdog_sota_achieved")
                return True

        # Step 4: Run auto-iteration
        _log("\n>>> Launching auto-iteration <<<")
        log_memory(f"watchdog_iteration_start:{iteration}")

        rc = run_auto_iterate()

        _log(f">>> Auto-iteration finished (rc={rc}) <<<")
        log_memory(f"watchdog_iteration_done:{iteration}:rc={rc}")

        # Brief pause before next check
        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Watchdog stopped by user")
    except Exception as e:
        _log(f"Watchdog error: {e}")
        import traceback
        traceback.print_exc()
