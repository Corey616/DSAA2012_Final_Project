"""
Standalone smoke test: verify SD3.5 Medium encoder + negative prompt support.
Runs directly against SD3.5, bypassing the StoryGen pipeline entirely.
"""
import os, sys, torch, time
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/lzz/DSAA2012_Final_Project/storygen/models"
os.environ.pop("HF_ENDPOINT", None)

OUT = Path("/home/lzz/DSAA2012_Final_Project/storygen/debug")
OUT.mkdir(exist_ok=True)
MODEL_ID = "adamo1139/stable-diffusion-3.5-medium-ungated"
CACHE = "/home/lzz/DSAA2012_Final_Project/storygen/models"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def is_gray(img):
    a = np.array(img.convert("RGB"))
    return a.std() < 5

def is_blank(img):
    a = np.array(img.convert("RGB"))
    return a.std() < 2

def test_single(name, **kwargs):
    """Run one generation and save. Returns (gray, blank, time)."""
    from diffusers import StableDiffusion3Pipeline
    if not hasattr(test_single, "_pipe"):
        log(f"Loading SD3.5...")
        test_single._pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, local_files_only=True, cache_dir=CACHE
        ).to("cuda:0")
        test_single._pipe.enable_model_cpu_offload()
    
    pipe = test_single._pipe
    base_kw = {
        "height": 512, "width": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "generator": torch.Generator("cuda:0").manual_seed(42),
    }
    base_kw.update(kwargs)
    
    t0 = time.time()
    try:
        img = pipe(**base_kw).images[0]
        elapsed = time.time() - t0
        gray = is_gray(img)
        blank = is_blank(img)
        a = np.array(img.convert("RGB"))
        img.save(OUT / f"{name}.png")
        log(f"  {name}: {img.size} mean={a.mean():.0f} std={a.std():.1f} gray={gray} blank={blank} ({elapsed:.1f}s)")
        return gray, blank, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  {name}: ERROR: {e}")
        return True, True, elapsed

# ── Test 1: Baseline txt2img ──
log("=== Test 1: Baseline (prompt only) ===")
g1, b1, _ = test_single("t1_baseline", prompt="a cat sitting on a sofa, anime style")

# ── Test 2: Negative prompt ──
log("=== Test 2: Negative prompt blocks 'dog' ===")
g2, b2, _ = test_single("t2_neg_block_dog",
    prompt="a cat sitting on a sofa, anime style",
    negative_prompt="dog, canine, puppy")

# ── Test 3: Negative prompt blocks style ──
log("=== Test 3: Negative blocks photorealistic ===")
g3, b3, _ = test_single("t3_neg_block_realistic",
    prompt="a cat sitting on a sofa, anime style",
    negative_prompt="photorealistic, 3D render, CGI, realistic photography")

# ── Test 4: prompt_2 (CLIP-G) ──
log("=== Test 4: prompt_2 (CLIP-G style anchor) ===")
g4, b4, _ = test_single("t4_prompt2_style",
    prompt="a cat sitting on a sofa",
    prompt_2="anime style, studio ghibli, cel shading, flat colors")

# ── Test 5: prompt_3 (T5-XXL) ──
log("=== Test 5: prompt_3 (T5-XXL long description) ===")
g5, b5, _ = test_single("t5_prompt3_t5",
    prompt="a cat sitting",
    prompt_3="a fluffy orange tabby cat sitting comfortably on a blue velvet sofa in a sunlit living room, anime style, studio ghibli, warm afternoon light")

# ── Test 6: All three encoders ──
log("=== Test 6: All three encoders ===")
g6, b6, _ = test_single("t6_all_three",
    prompt="a cat on a sofa",
    prompt_2="anime style, studio ghibli, clean lineart, cel shading",
    prompt_3="a fluffy orange tabby cat sitting comfortably on a blue velvet sofa, warm afternoon sunlight streaming through window, cozy living room atmosphere, anime style")

# ── Test 7: max_sequence_length with prompt_3 ──
log("=== Test 7: max_sequence_length=256 ===")
g7, b7, _ = test_single("t7_maxseq",
    prompt="a cat sitting",
    prompt_3="a fluffy orange tabby cat sitting comfortably on a blue velvet sofa in a sunlit living room with bookshelves and plants, anime style, studio ghibli, warm afternoon light streaming through large windows, cozy atmosphere, detailed background",
    max_sequence_length=256)

# ── Report ──
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
results = [
    ("T1: Baseline",             g1, b1, "CLIP-L only"),
    ("T2: Neg blocks 'dog'",     g2, b2, "negative_prompt works?"),
    ("T3: Neg blocks realistic", g3, b3, "negative_prompt style?"),
    ("T4: prompt_2 (CLIP-G)",    g4, b4, "CLIP-G supported?"),
    ("T5: prompt_3 (T5-XXL)",    g5, b5, "T5-XXL supported?"),
    ("T6: All 3 encoders",       g6, b6, "triple encoder?"),
    ("T7: max_sequence_length",  g7, b7, "max_seq with T5?"),
]
for name, gray, blank, note in results:
    status = "❌ GRAY" if gray else ("⚠️ BLANK" if blank else "✅ OK")
    print(f"  {status:<12} {name:<30} {note}")

print(f"\nImages saved to: {OUT}/")
