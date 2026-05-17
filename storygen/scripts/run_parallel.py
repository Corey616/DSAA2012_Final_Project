#!/usr/bin/env python3
"""
Multi-GPU parallel batch runner for TaskA story generation.
Distributes 32 test cases across available GPUs (≥30GB free each).
Usage: python run_parallel.py [--max-gpus N] [--output outputs/parallel_batch]
"""
import subprocess
import json
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


def get_free_gpus(min_free_mb: int = 30000) -> list[int]:
    """Query nvidia-smi and return GPUs with ≥min_free_mb MB free."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            idx = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if free_mb >= min_free_mb:
                gpus.append(idx)
        return gpus
    except Exception as e:
        print(f"[GPU] Failed to query GPUs: {e}")
        return []


def run_worker(gpu_id: int, script_path: str, output_dir: str) -> dict:
    """Launch a worker subprocess on a specific GPU."""
    worker_script = Path(__file__).absolute().parent / "run_worker.py"
    env = os.environ.copy()

    cmd = [
        sys.executable, str(worker_script),
        "--gpu", str(gpu_id),
        "--script", script_path,
        "--output", output_dir,
    ]

    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=Path(__file__).absolute().parent.parent,
        timeout=600,  # 10 min per case
    )
    elapsed = time.time() - t0

    result = {"gpu": gpu_id, "script": script_path, "elapsed_s": round(elapsed, 1)}

    if proc.returncode != 0:
        result["status"] = "worker_error"
        result["error"] = proc.stderr[-500:] if proc.stderr else f"exit code {proc.returncode}"
        return result

    for line in proc.stdout.split("\n"):
        if line.startswith("WORKER_DONE|"):
            parts = line.split("|")
            result["status"] = "success"
            result["case_name"] = parts[3]
            result["metrics"] = parts[5] if len(parts) > 5 else ""
            return result

    result["status"] = "timeout_or_parse_error"
    result["stdout_tail"] = proc.stdout[-300:] if proc.stdout else ""
    result["stderr_tail"] = proc.stderr[-300:] if proc.stderr else ""
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU parallel TaskA batch runner")
    parser.add_argument("--max-gpus", type=int, default=8, help="Max GPUs to use")
    parser.add_argument("--output", type=str, default="outputs/parallel_batch")
    parser.add_argument("--min-free-gb", type=int, default=30, help="Min free GPU memory in GB")
    parser.add_argument("--cases", type=str, nargs="*", help="Specific cases (e.g., 01 03 06)")
    args = parser.parse_args()

    min_free_mb = args.min_free_gb * 1024

    gpus = get_free_gpus(min_free_mb)
    if args.max_gpus:
        gpus = gpus[: args.max_gpus]

    if not gpus:
        print("❌ No GPUs with sufficient free memory. Aborting.")
        sys.exit(1)

    print(f"🖥️  Available GPUs ({len(gpus)}): {gpus}")
    print(f"   Min free memory: {args.min_free_gb} GB\n")

    taska_dir = Path("data/TaskA")
    if args.cases:
        cases = [taska_dir / f"{c}.txt" for c in args.cases]
    else:
        cases = sorted(taska_dir.glob("*.txt"))

    print(f"📂 {len(cases)} test cases to process\n")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, case in enumerate(cases):
        gpu = gpus[i % len(gpus)]
        case_output = str(output_base / case.stem)
        tasks.append((gpu, str(case), case_output))

    print(f"🚀 Launching {len(tasks)} workers across {len(gpus)} GPUs...\n")
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_worker, gpu, script, out): (gpu, script)
            for gpu, script, out in tasks
        }
        for future in as_completed(futures):
            gpu, script = futures[future]
            try:
                r = future.result()
                results.append(r)
                status = r.get("status", "unknown")
                name = Path(script).stem
                icon = "✅" if status == "success" else "❌"
                metrics = r.get("metrics", "")
                print(f"  {icon} GPU{gpu} | {name:12s} | {status:8s} | {metrics}")
            except Exception as e:
                results.append({"gpu": gpu, "script": script, "status": "exception", "error": str(e)})
                print(f"  ❌ GPU{gpu} | {Path(script).stem:12s} | exception | {e}")

    total_time = time.time() - start_time
    success = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - success

    print(f"\n{'='*70}")
    print(f"📊 BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"  GPUs used:     {len(gpus)}")
    print(f"  Total cases:   {len(cases)}")
    print(f"  Successful:    {success}")
    print(f"  Failed:        {failed}")
    print(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    if success > 0:
        print(f"  Avg per case:  {total_time/success:.1f}s")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "gpus": gpus,
        "total_cases": len(cases),
        "success": success,
        "failed": failed,
        "total_time_s": round(total_time, 1),
        "results": results,
    }
    summary_path = output_base / "parallel_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
