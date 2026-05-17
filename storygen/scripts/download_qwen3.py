#!/usr/bin/env python3
"""Download Qwen/Qwen3-4B-Instruct-2507 into the project model cache."""
import sys
import os
import time
from pathlib import Path

# Explicit project cache path (bypass editable install issue)
PROJECT_CACHE_DIR = Path("/home/lzz/DSAA2012_Final_Project/storygen/models")

start_time = time.time()

print("=" * 70)
print("Downloading Qwen/Qwen3-4B-Instruct-2507")
print("=" * 70)

# Step 0: Configure env vars directly
os.environ["HF_HOME"] = str(PROJECT_CACHE_DIR)
os.environ["TORCH_HOME"] = str(PROJECT_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(PROJECT_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_CACHE_DIR)
os.environ.pop("HF_ENDPOINT", None)  # Direct HuggingFace (not mirror)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nCache directory: {PROJECT_CACHE_DIR}")
print(f"Cache directory exists: {PROJECT_CACHE_DIR.exists()}")

# Step 0.5: Clean any previous incomplete downloads
print("\n[Step 0] Cleaning any previous incomplete downloads...")
# Custom cleanup for our cache dir
import shutil
for model_dir in PROJECT_CACHE_DIR.glob("models--*"):
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists():
        for blob_file in blobs_dir.iterdir():
            try:
                if blob_file.suffix == '.incomplete' or blob_file.stat().st_size == 0:
                    blob_file.unlink()
                    print(f"  Removed incomplete: {blob_file.name}")
            except (FileNotFoundError, OSError):
                pass
print("  Cleanup complete.")

# Step 1: Check if model already cached
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
print(f"\n[Step 1] Checking if {MODEL_NAME} already cached...")

model_cache_path = PROJECT_CACHE_DIR / f"models--{MODEL_NAME.replace('/', '--')}"

# Manual integrity check
def check_integrity():
    if not model_cache_path.exists():
        return False
    blobs_dir = model_cache_path / "blobs"
    if blobs_dir.exists():
        for f in blobs_dir.iterdir():
            try:
                if f.suffix == '.incomplete' or f.stat().st_size == 0:
                    return False
            except FileNotFoundError:
                return False
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False
    snapshot_dirs = list(snapshots_dir.iterdir())
    if len(snapshot_dirs) == 0:
        return False
    snapshot_dir = snapshot_dirs[0]
    has_model_files = any(
        f.suffix in ('.safetensors', '.bin', '.json', '.msgpack', '.h5', '.ot', '.pth')
        for f in snapshot_dir.iterdir()
    )
    return has_model_files

if check_integrity():
    total_size = sum(f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()) / (1024**3)
    print(f"  Model already exists and is COMPLETE ({total_size:.2f} GB). Nothing to do.")
    sys.exit(0)
else:
    print("  Model not cached or incomplete. Proceeding with download...")

# Step 2: Download
print(f"\n[Step 2] Downloading {MODEL_NAME} (~8GB)...")
print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Cache dir: {PROJECT_CACHE_DIR}")
print(f"  This may take a while depending on network speed...\n")

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    MODEL_NAME,
    cache_dir=str(PROJECT_CACHE_DIR),
    resume_download=True,
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
)

download_time = time.time() - start_time

# Step 3: Verify
print(f"\n[Step 3] Verifying model integrity...")
is_complete = check_integrity()

# Step 4: Get size
total_size = sum(f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()) / (1024**3)
size_gb = round(total_size, 2)

# Step 5: Report
print("\n" + "=" * 70)
print("DOWNLOAD REPORT")
print("=" * 70)
print(f"  Model:         {MODEL_NAME}")
print(f"  Cache path:    {model_cache_path}")
print(f"  Size:          {size_gb} GB")
print(f"  Download time: {download_time:.1f} seconds ({download_time/60:.1f} minutes)")
print(f"  Integrity:     {'COMPLETE' if is_complete else 'INCOMPLETE'}")
print(f"  Downloaded to: {model_path}")
print("=" * 70)

# Step 6: List cached models
print("\n[Step 4] All cached Qwen models:")
for model_dir in sorted(PROJECT_CACHE_DIR.glob("models--Qwen--*")):
    name = model_dir.name.replace("models--", "").replace("--", "/")
    sz = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024**3)
    print(f"  {name:45s} {sz:6.2f} GB")

if not is_complete:
    print("\nWARNING: Model integrity check FAILED.")
    sys.exit(1)
else:
    print(f"\nQwen3-4B-Instruct-2507 is ready to use!")
    print(f"Cache path: {model_cache_path}")
