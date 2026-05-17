# StoryGen — Narrative Story Visualization

Generate multi-frame stories from text scripts using SD3.5 + Qwen3.

## Prerequisites

- **Conda** (Miniforge / Miniconda recommended) — [install guide](https://github.com/conda-forge/miniforge)
- **NVIDIA GPU** with ≥80 GB VRAM (tested on A800 80 GB); CUDA 12.4+ driver
- **~80 GB free disk space** (models: ~60 GB, conda+pip packages: ~10 GB, output cache: ~10 GB)
- **Linux x86_64** (other platforms not tested; CPU-only fallback possible but extremely slow)

## Quick Start

### 1. Clone the repo

```bash
git clone <this-repo-url> storygen
cd storygen/storygen
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml -p ./env
```

> **What this does:** Creates a conda environment at `./env` with Python 3.12 + all pip dependencies (torch, diffusers, transformers, open-clip-torch, etc.).
>
> **Note:** The environment.yml intentionally uses relaxed version pins (e.g. `torch>=2.4,<3.0`). This maximizes compatibility across different CUDA driver versions. The first `pip install` inside the environment will resolve and download the correct CUDA variant automatically.

### 3. Download the models

The pipeline requires four pretrained models from HuggingFace Hub. The code automatically caches them to `storygen/models/` on first use.

```bash
# Activate environment
conda activate ./env

# Run a small script to trigger downloads (models will auto-download on first run):
python scripts/run_worker.py --gpu 0 --script data/TaskA/01.txt --output outputs/first_run
```

**Automatic model downloads will include:**

| Model | Disk | Purpose |
|---|---|---|
| `adamo1139/stable-diffusion-3.5-medium-ungated` | ~16 GB | Main image generation |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~31 GB | Character portrait generation |
| `Qwen/Qwen3-4B-Instruct-2507` | ~7.6 GB | Script → storyboard parsing |
| `h94/IP-Adapter` | ~24 GB | Identity consistency |

**Cache directory:** All models are stored in `storygen/models/` following HuggingFace Hub cache layout (`models--org--name/`). The environment variable overrides are set automatically at runtime by `src/utils/mirror_config.py`.

> **Offline / air-gapped setup:** Download the models on an internet-connected machine, then copy the `models/` directory into the project:
> ```bash
> # On an internet-connected machine:
> export HF_HOME=./models HF_HUB_CACHE=./models
> python -c "from huggingface_hub import snapshot_download; snapshot_download('adamo1139/stable-diffusion-3.5-medium-ungated')"
> # Repeat for: stabilityai/stable-diffusion-xl-base-1.0, Qwen/Qwen3-4B-Instruct-2507, h94/IP-Adapter
>
> # Copy the entire models/ dir to the target machine
> ```

### 4. Run a test case

```bash
conda run -p ./env python scripts/run_worker.py --gpu 0 --script data/TaskA/01.txt --output outputs/test_run
```

Expected output (approx. 4 minutes on A800 80 GB):

```
WORKER_DONE|✅|GPU0|01.txt|success|CLIP=0.157 CONS=0.414 OVERALL=0.260
```

### 5. Verify the output

```bash
ls outputs/test_run/
# frame_01.png  frame_02.png  frame_03.png  production_board.json  result.json  storyboard.png
python -c "
from PIL import Image; import numpy as np, os
d = 'outputs/test_run'
for f in sorted(os.listdir(d)):
    if f.endswith('.png'):
        img = Image.open(os.path.join(d, f)).convert('RGB')
        a = np.array(img)
        is_blank = a.std() < 2
        print(f'{f}: {a.shape} mean={a.mean():.1f} std={a.std():.1f} blank={is_blank}')
"
```

Non-blank images have `std > 10`. Expected: ~70–90 for valid colorful images.

## Project Structure

```
storygen/
├── environment.yml         # Conda environment (this file)
├── README.md               # This file
├── requirements.txt        # pip freeze (reference only)
├── models/                 # HuggingFace Hub cache (auto-populated)
├── data/
│   └── TaskA/              # Example scripts
│       ├── 01.txt          # 3-panel story
│       ├── 02.txt
│       └── ...
├── scripts/
│   ├── run_worker.py       # Single-GPU single-script runner
│   └── run_parallel.py     # Multi-GPU batch runner
├── outputs/                # Generated images + metrics
├── src/
│   ├── core_generator/     # SD3.5 pipeline + attention processors
│   ├── script_director/    # Qwen3-based script parser
│   ├── asset_anchor/       # Character portrait generation
│   ├── evaluation_hub/     # CLIP + LPIPS metric evaluators
│   └── utils/              # Mirror config, image utils
└── env/                    # Conda environment (created by setup)
```

## Usage

### Running a single script

```bash
python scripts/run_worker.py --gpu 0 --script data/TaskA/01.txt --output outputs/my_story
```

Outputs (per run):
- `frame_XX.png` — Individual story frames (1024×1024)
- `storyboard.png` — Side-by-side montage
- `production_board.json` — Parsed script + character descriptions
- `result.json` — Evaluation metrics (CLIP, LPIPS, overall score)

### Writing custom scripts

Scripts use a simple text format:

```
[SCENE-1] <CharacterName> does something in the setting.

[SEP]

[SCENE-2] <CharacterName> does something else.
```

- Lines starting with `[SCENE-N]` define each panel
- `<CharacterName>` tags identify characters (angle brackets)
- `[SEP]` separates panels
- The LLM parser (Qwen3) automatically:
  - Generates character visual descriptions
  - Enriches scene prompts with style, lighting, and object details
  - Determines shot types (close-up, medium, wide)

Place custom `.txt` files in `data/TaskA/` and run with `--script data/TaskA/your_script.txt`.

### Running batch evaluation

```bash
python scripts/run_parallel.py --gpu 0,1,2,3 --task-a
```

Distributes data/TaskA/*.txt scripts across available GPUs.

## Output Metrics

Metrics saved in `result.json`:

| Metric | Range | Meaning |
|---|---|---|
| `avg_clip_score` | 0–1 | Text-image alignment (higher = better) |
| `avg_consistency` | 0–1 | Frame-to-frame coherence (lower LPIPS = higher) |
| `overall_score` | 0–1 | Weighted: 0.6×CLIP + 0.4×consistency |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | GPU VRAM < 80 GB | Set `enable_model_cpu_offload: true` in config |
| `ModuleNotFoundError` | Conda env not activated | `conda activate ./env` |
| All images are gray/blank | Model not loaded (cache miss) | Check `models/` dir has expected `models--*` folders |
| Qwen parsing fails | Corrupted Qwen download | Delete `models/models--Qwen--Qwen3-4B-Instruct-2507/` and re-run |
| `No such file or directory: '.../work'` | Paths from pip freeze broken | Ignore `requirements.txt`; use `environment.yml` for setup |

## Advanced: Custom Conda Environment Name

Instead of a prefix-based environment, you can use a named env:

```bash
conda env create -f environment.yml -n storygen
conda activate storygen
python scripts/run_worker.py --gpu 0 --script data/TaskA/01.txt --output outputs/test_run
```

## License

Internal academic use. See course materials for details.
