#!/usr/bin/env python3
# File: src/sdxl_generator.py
# Description: Baseline SDXL 1.0 Image Generator.
# This script strictly follows the "Baseline Inference" requirement:
# - Offline operation (no internet connection required after download)
# - Local model loading
# - Simple text-to-image generation

import os
import sys
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
import torch

# ---------------------------------------------------
# 🔒 安全设置：禁止任何网络请求 (Compliance Requirement)
# ---------------------------------------------------
# 为了满足项目合规性（禁止Agent/外部API），我们强制 diffusers 离线运行。
# 如果模型文件已下载，这将确保代码不会尝试连接 Hugging Face Hub。
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 可选：如果你的环境中没有网络，取消下面这行的注释，强制 PyTorch 也不尝试下载预训练权重
# os.environ['TORCH_HOME'] = '/path/to/your/local/torch/cache' 

class SDXLGenerator:
    """
    A wrapper class for Stable Diffusion XL 1.0 baseline generation.
    This meets the "Baseline Inference" checkpoint requirement.
    """
    
    def __init__(self, model_path: str = None, device='cuda'):
        """
        Initialize the pipeline.
        
        Args:
            model_path (str): Path to the local SDXL model directory.
                             If None, it defaults to 'models/sdxl' as per the project plan.
        """
        # 1. 设置默认路径 (符合项目架构映射)
        if model_path is None:
            self.model_path = Path(__file__).parent.parent / "models/sdxl/sd_xl_base_1.0.safetensors"
        else:
            self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}. "
                                   "Please run scripts/download_models.sh first.")
        
        print(f"📂 Loading SDXL from: {self.model_path}")
        
        # 2. 构建流水线 (Pipeline)
        # 注意: 我们在这里显式指定 local_files_only=True 以确保合规
        try:
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path.as_posix(),
                torch_dtype=torch.float16, # 推荐使用半精度以节省显存
                local_files_only=True # 🚫 关键合规设置：绝对禁止网络请求
            )
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Ensure the model is correctly downloaded in the 'models/sdxl' directory.")
            raise
        
        # 3. 设备选择 (GPU优先)
        # 根据文档，我们假设环境已配置好 CUDA (environment.yml)
        self.device = device
        if self.device == "cpu":
            print("⚠️  Warning: CUDA not available. Falling back to CPU (very slow).")
        
        self.pipe.to(self.device)
        
        # 4. ⚠️ 安全措施：如果使用 CPU，限制高度和宽度以防止内存爆炸
        if self.device == "cpu":
            self.pipe.enable_attention_slicing()
            print("Enabled attention slicing for CPU memory efficiency.")

    def generate_image(self, prompt: str, output_path: str, 
                       negative_prompt: str = None,
                       width: int = 1024, height: int = 1024,
                       guidance_scale: float = 7.5, num_inference_steps: int = 30) -> bool:
        """
        Generate a single image from a text prompt.
        
        This function satisfies the "Baseline Inference" requirement in the plan.
        
        Args:
            prompt (str): The text prompt describing the image.
            output_path (str): Where to save the generated image (e.g., 'results/test_cat.png').
            negative_prompt (str, optional): Things to avoid in the image.
            width (int): Image width.
            height (int): Image height.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"🎨 Generating image for prompt: {prompt[:50]}...") # 打印前50个字符
        print(f"💾 Output path: {output_path}")
        
        try:
            # 执行推理
            # 注意: 对于 SDXL，建议分辨率是 1024x1024
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, low quality, text, watermark",
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # SDXL 需要指定生成的高宽，这对多图生成的一致性很重要
            ).images[0]
            
            # 保存图像
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            image.save(output_path)
            print(f"✅ Success! Image saved to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False


# ---------------------------------------------------
# 🚀 Main Block: "Hello World" Test
# ---------------------------------------------------
# 这个部分用于验证 Checkpoint 1.3: 基线推理
# 运行此脚本直接生成测试图
if __name__ == "__main__":
    """
    This main block serves as the "Hello World" test.
    It verifies that the environment is set up correctly.
    """
    print("🚀 Starting SDXL Baseline Test...")
    
    # 1. 初始化生成器
    try:
        generator = SDXLGenerator()
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Please check your model path and environment.yml dependencies.")
        sys.exit(1)
    
    # 2. 定义测试参数 (Hard-coded for the baseline test)
    test_prompt = "A photo of a cute cat sitting on a windowsill, sunny day, realistic"
    test_output = "results/test_cat.png" # 符合计划中的验证路径
    
    # 3. 执行生成
    success = generator.generate_image(
        prompt=test_prompt,
        output_path=test_output,
        width=1024,
        height=1024
    )
    
    if success:
        print("\n🎉 Baseline Test Passed! You are ready for the next stage.")
        print("Next Step: Proceed to LLM Prompt Engineering (Stage 2).")
    else:
        print("\n🛑 Baseline Test Failed. Check the error logs above.")
        sys.exit(1)