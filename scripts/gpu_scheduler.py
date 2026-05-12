#!/usr/bin/env python3
"""
GPU Scheduler — Dynamic GPU allocation for autonomous iteration.
Features:
- Query GPU memory via nvidia-smi
- File-based locks for cross-process safety
- Wait with timeout and backoff
- Orphan lock cleanup
- Usage logging to memory
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

LOCK_DIR = Path("/tmp/storygen_gpu_locks")

# GPU requirements per task type (MB)
GPU_REQUIREMENTS = {
    "sdxl_generate": 20000,    # SDXL full GPU (no offload)
    "sdxl_offload": 8000,      # SDXL with CPU offload (~6-8GB)
    "clip_eval": 2048,          # CLIP evaluation
    "lpips_eval": 2048,         # LPIPS consistency
    "qwen_parse": 2048,         # Qwen2.5 parser (~2GB)
    "vlm_diag": 4096,           # VLM diagnostic
}


def _parse_nvidia_smi() -> list[dict]:
    """Parse nvidia-smi to get GPU memory info."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "memory_free": int(parts[1]),
                    "memory_used": int(parts[2]),
                    "memory_total": int(parts[3]),
                    "name": parts[4],
                })
        return gpus
    except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
        print(f"[GPUScheduler] nvidia-smi error: {e}")
        return []


def _cleanup_orphan_locks():
    """Remove locks for dead processes."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for lock_file in LOCK_DIR.glob("gpu_*.lock"):
        try:
            data = json.loads(lock_file.read_text())
            pid = data.get("pid", 0)
            # Check if process is alive
            try:
                os.kill(pid, 0)  # Signal 0 = test existence
            except (OSError, ProcessLookupError):
                lock_file.unlink(missing_ok=True)
                print(f"[GPUScheduler] Cleaned orphan lock: {lock_file.name}")
        except (json.JSONDecodeError, OSError):
            lock_file.unlink(missing_ok=True)


def _get_locked_gpus() -> set[int]:
    """Get set of GPUs that have active locks."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    locked = set()
    for lock_file in LOCK_DIR.glob("gpu_*.lock"):
        try:
            data = json.loads(lock_file.read_text())
            pid = data.get("pid", 0)
            try:
                os.kill(pid, 0)
                locked.add(data["gpu_id"])
            except (OSError, ProcessLookupError):
                lock_file.unlink(missing_ok=True)
        except (json.JSONDecodeError, KeyError):
            lock_file.unlink(missing_ok=True)
    return locked


def get_available_gpu(
    min_memory_mb: int = 20000,
    exclude_gpus: Optional[set[int]] = None,
) -> Optional[int]:
    """
    Get an available GPU with at least min_memory_mb free.
    Returns GPU index or None if none available.
    """
    _cleanup_orphan_locks()
    locked = _get_locked_gpus()
    exclude = exclude_gpus or set()
    gpus = _parse_nvidia_smi()

    for gpu in sorted(gpus, key=lambda x: -x["memory_free"]):
        idx = gpu["index"]
        if idx in locked:
            continue
        if idx in exclude:
            continue
        if gpu["memory_free"] >= min_memory_mb:
            return idx
    return None


def acquire_gpu(
    min_memory_mb: int = 20000,
    exclude_gpus: Optional[set[int]] = None,
    holder: str = "unknown",
) -> Optional[int]:
    """
    Atomically acquire a GPU lock.
    Returns GPU index or None if none available.
    """
    _cleanup_orphan_locks()
    locked = _get_locked_gpus()
    exclude = exclude_gpus or set()
    gpus = _parse_nvidia_smi()

    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    for gpu in sorted(gpus, key=lambda x: -x["memory_free"]):
        idx = gpu["index"]
        if idx in locked:
            continue
        if idx in exclude:
            continue
        if gpu["memory_free"] < min_memory_mb:
            continue

        # Atomically create lock file
        lock_path = LOCK_DIR / f"gpu_{idx}.lock"
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                json.dump({
                    "gpu_id": idx,
                    "pid": os.getpid(),
                    "holder": holder,
                    "acquired_at": time.time(),
                    "memory_free_at_acquire": gpu["memory_free"],
                }, f)
            print(f"[GPUScheduler] 🔒 GPU {idx} acquired by {holder} "
                  f"(free: {gpu['memory_free']}MB, need: {min_memory_mb}MB)")
            return idx
        except FileExistsError:
            continue  # Race: another process got it first

    return None


def wait_for_gpu(
    min_memory_mb: int = 20000,
    poll_interval: int = 30,
    timeout: int = 3600,
    exclude_gpus: Optional[set[int]] = None,
    holder: str = "unknown",
) -> int:
    """
    Block until a GPU with sufficient memory is available.
    Raises TimeoutError if timeout reached.
    """
    start = time.time()
    waited = 0
    while time.time() - start < timeout:
        gpu = acquire_gpu(min_memory_mb, exclude_gpus, holder)
        if gpu is not None:
            if waited > 0:
                print(f"[GPUScheduler] Waited {waited}s for GPU")
            return gpu

        time.sleep(poll_interval)
        waited += poll_interval
        if waited % 120 == 0:  # Log every 2 minutes
            print(f"[GPUScheduler] ⏳ Waiting for GPU ({min_memory_mb}MB free) "
                  f"for {holder}... ({waited}s elapsed)")

    raise TimeoutError(
        f"[GPUScheduler] Timed out after {timeout}s waiting for GPU "
        f"with {min_memory_mb}MB free for {holder}"
    )


def release_gpu(gpu_id: int):
    """Release a GPU lock."""
    lock_path = LOCK_DIR / f"gpu_{gpu_id}.lock"
    try:
        data = json.loads(lock_path.read_text())
        if data.get("pid") == os.getpid():
            lock_path.unlink()
            print(f"[GPUScheduler] 🔓 GPU {gpu_id} released by {data.get('holder', '?')}")
        else:
            print(f"[GPUScheduler] ⚠️ GPU {gpu_id} lock belongs to PID {data.get('pid')}, "
                  f"not releasing (our PID: {os.getpid()})")
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # No lock exists


def release_all_gpus(holder: Optional[str] = None):
    """Release all GPU locks held by current process (or by a specific holder)."""
    for lock_file in LOCK_DIR.glob("gpu_*.lock"):
        try:
            data = json.loads(lock_file.read_text())
            pid_match = data.get("pid") == os.getpid()
            holder_match = holder is None or data.get("holder") == holder
            if pid_match or holder_match:
                lock_file.unlink()
                print(f"[GPUScheduler] 🔓 Released GPU {data['gpu_id']} "
                      f"(holder: {data.get('holder', '?')})")
        except (json.JSONDecodeError, FileNotFoundError):
            pass


def print_status():
    """Print current GPU status with lock info."""
    gpus = _parse_nvidia_smi()
    locked = _get_locked_gpus()
    print(f"{'GPU':>4} {'Name':20} {'Total':>8} {'Free':>8} {'Used':>8} {'Lock':>6}")
    print("-" * 60)
    for gpu in gpus:
        lock_status = "🔒" if gpu["index"] in locked else "  "
        print(f"{gpu['index']:>4} {gpu['name']:20} "
              f"{gpu['memory_total']:>8} {gpu['memory_free']:>8} "
              f"{gpu['memory_used']:>8} {lock_status:>6}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print_status()
    elif len(sys.argv) > 2 and sys.argv[1] == "acquire":
        mem = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        gpu = acquire_gpu(mem, holder=f"cli_{os.getpid()}")
        if gpu is not None:
            print(f"GPU {gpu}")
        else:
            print("No GPU available", file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) > 2 and sys.argv[1] == "release":
        release_gpu(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "release-all":
        release_all_gpus()
    else:
        print(f"Usage: {sys.argv[0]} status|acquire <MB>|release <GPU_ID>|release-all")
