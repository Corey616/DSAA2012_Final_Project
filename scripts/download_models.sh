#!/bin/bash

# File: scripts/download_models_incremental.sh
# Description: Incremental download script that skips existing files.

echo "🚀 Starting incremental download of Project 2 model weights..."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "📁 Project Root: $ROOT_DIR"
cd "$ROOT_DIR/models" || exit 1

# 错误计数器（用于后台任务）
ERRORS=0
ERROR_LOCK=$(mktemp -u)

# 线程安全的错误计数增加函数
increment_error() {
    # 简单的文件锁机制防止竞态条件
    (
        flock -x 200
        ERRORS=$((ERRORS + 1))
    ) 200>"$ERROR_LOCK"
}

# ==========================================
# Helper: Parallel Download (only if file doesn't exist)
# ==========================================
download_async_if_missing() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "⏭️  [SKIP] $output (already exists)"
        return 0 # 文件已存在，不计入错误
    fi
    echo "🔽 [ASYNC] $output"
    # -x 8 足够了，因为多个文件同时多线程，总数会很大
    aria2c -x 8 -s 8 -c --quiet "$url" -o "$output"
    # 检查返回值
    if [ $? -ne 0 ]; then
        echo "❌ Failed: $output"
        increment_error
    fi
}

# ==========================================
# 1. Start SDXL Download (Background)
# ==========================================
echo "📦 [1/3] Queuing SDXL..."
SDXL_DIR="sdxl"
mkdir -p "$SDXL_DIR"
(
    cd "$SDXL_DIR" || exit 1
    # Config
    if [ ! -f "model_index.json" ]; then
        echo "🔽 [CONFIG] model_index.json"
        aria2c -x 4 -c --quiet "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json" -o "model_index.json" || increment_error
    fi
    # Checkpoint
    download_async_if_missing "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" "sd_xl_base_1.0.safetensors"
) &

# ==========================================
# 2. IP-Adapter Download with Exact Pattern
# ==========================================
download_ipadapter_exact_pattern() {
    echo "📦 [2/3] Downloading IP-Adapter by exact pattern: sdxl_models/*.safetensors, sdxl_models/image_encoder/*"
    
    IPADAPTER_DIR="ip-adapter"
    mkdir -p "$IPADAPTER_DIR"
    cd "$IPADAPTER_DIR" || { 
        echo "❌ Cannot enter $IPADAPTER_DIR"
        increment_error
        return 1
    }
    
    BASE_URL="https://hf-mirror.com/h94/IP-Adapter/resolve/main"
    
    # 创建下载列表文件
    INPUT_FILE="ipadapter_download_list.txt"
    
    # 根据你的规则，我们需要先获取文件列表
    echo "🔍 Fetching file list from HuggingFace..."
    
    # 方法1：使用 huggingface-hub Python 库（如果可用）
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 << 'EOF'
import os
from huggingface_hub import HfApi, hf_hub_url
import sys

api = HfApi()
files = api.list_repo_files(repo_id="h94/IP-Adapter", repo_type="model")

# 过滤出符合规则的文件
pattern1 = "sdxl_models/"
pattern2 = "models/image_encoder/"

for file in files:
    if (file.startswith("sdxl_models/") and file.endswith(".safetensors")) or \
       file.startswith("models/image_encoder/"):
        # 构造下载URL
        url = f"https://hf-mirror.com/h94/IP-Adapter/resolve/main/{file}"
        print(url)
EOF
    else
        # 方法2：硬编码已知文件
        echo "⚠️ huggingface_hub not available, using hardcoded list"
        cat << 'EOF'
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin
EOF
    fi > "$INPUT_FILE"
    
    # 检查文件数量
    FILE_COUNT=$(wc -l < "$INPUT_FILE" 2>/dev/null || echo "0")
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo "❌ Failed to get file list"
        # 回退到手动列表
        cat > "$INPUT_FILE" << 'EOF'
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/config.json
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/preprocessor_config.json
https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
EOF
        FILE_COUNT=$(wc -l < "$INPUT_FILE")
    fi
    
    echo "📄 Found $FILE_COUNT files matching pattern"
    
    # 使用 aria2c 批量下载
    echo "🚀 Starting aria2c batch download..."
    aria2c \
        -i "$INPUT_FILE" \
        -j 3 \
        -x 8 \
        -s 8 \
        -c \
        --auto-file-renaming=false \
        --conditional-get=true \
        --file-allocation=none \
        --summary-interval=0 \
        --max-tries=5 \
        --retry-wait=3 \
        --connect-timeout=10 \
        --timeout=30 \
        --quiet=false
    
    if [ $? -ne 0 ]; then
        echo "⚠️ Some downloads may have failed"
    fi
    
    # 验证下载的文件
    echo "🔍 Verifying downloaded files..."
    MISSING=0
    while IFS= read -r url; do
        if [ -z "$url" ]; then continue; fi
        
        # 提取文件名
        filename=$(echo "$url" | sed 's|.*/||')
        dirpath=$(echo "$url" | sed -n 's|.*IP-Adapter/resolve/main/||p' | sed "s|/$filename||")
        
        if [ -n "$dirpath" ] && [ "$dirpath" != "$filename" ]; then
            filepath="$dirpath/$filename"
        else
            filepath="$filename"
        fi
        
        if [ ! -f "$filepath" ]; then
            echo "❌ Missing: $filepath"
            MISSING=$((MISSING + 1))
        else
            filesize=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo "0")
            if [ "$filesize" -lt 1000 ]; then
                echo "⚠️  Suspiciously small: $filepath ($filesize bytes)"
            else
                echo "✅ OK: $filepath"
            fi
        fi
    done < "$INPUT_FILE"
    
    if [ "$MISSING" -gt 0 ]; then
        echo "⚠️  $MISSING files are missing"
    else
        echo "✅ All files downloaded successfully"
    fi
    
    # 清理
    rm -f "$INPUT_FILE"
    
    cd "$ROOT_DIR/models" || return
}
download_ipadapter_exact_pattern

# ==========================================
# 3. Improved LLM Download (串行小文件 + 并行大文件) with Skip Logic
# ==========================================
echo "📦 [3/3] Downloading LLM (Qwen2.5 7B)..."
LLM_DIR="llm/Qwen2.5-7B-Instruct"
mkdir -p "$LLM_DIR"
cd "$LLM_DIR" || exit 1

BASE_URL="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main"

# --- Key Improvement: Check for critical config files first ---
CRITICAL_CONFIGS=("config.json" "tokenizer_config.json")
MISSING_CONFIGS=()
for conf in "${CRITICAL_CONFIGS[@]}"; do
    if [ ! -f "$conf" ]; then
        MISSING_CONFIGS+=("$conf")
    fi
done

if [ ${#MISSING_CONFIGS[@]} -gt 0 ]; then
    echo "📝 [NEED CONFIG] Downloading missing config/tokenizer files: ${MISSING_CONFIGS[*]}"
    # Define all required small files
    SMALL_FILES=(
        "config.json"
        "generation_config.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
        "model.safetensors.index.json"
    )

    # Iterate and download only missing ones
    for file in "${SMALL_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "🔽 [SERIAL] $file"
            aria2c -x 4 -c --quiet "$BASE_URL/$file" -o "$file"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to download small file: $file"
                increment_error
            fi
        else
            echo "⏭️  [SKIP] $file (already exists)"
        fi
    done
else
    echo "⏭️  [SKIP] All critical config files already exist, proceeding to shards."
fi

# --- Check for model shards ---
SHARDS=(
    "model-00001-of-00004.safetensors"
    "model-00002-of-00004.safetensors"
    "model-00003-of-00004.safetensors"
    "model-00004-of-00004.safetensors"
)

MISSING_SHARDS=()
for shard in "${SHARDS[@]}"; do
    if [ ! -f "$shard" ]; then
        MISSING_SHARDS+=("$shard")
    fi
done

if [ ${#MISSING_SHARDS[@]} -gt 0 ]; then
    echo "🚀 [NEED SHARDS] Downloading missing model shards: ${MISSING_SHARDS[*]}"
    # Download remaining shards in parallel
    for shard in "${SHARDS[@]}"; do
        download_async_if_missing "${BASE_URL}/${shard}" "$shard"
    done
else
    echo "⏭️  [SKIP] All model shards already exist."
fi

# ==========================================
# 4. Wait for all tasks and finalize
# ==========================================
echo "⏳ Waiting for all downloads to complete... (This may take a while)"
wait

# 清理锁文件
rm -f "$ERROR_LOCK"

# 检查是否有错误
if [ $ERRORS -gt 0 ]; then
    echo "❌ CRITICAL: Download completed with $ERRORS errors. Please check logs."
    exit 1
else
    echo "✅ ALL downloads completed successfully or skipped (if already present)! Enjoy your models."
fi