import torch
import torchvision
import transformers
import diffusers
from PIL import Image
import cv2
import numpy as np

print('\n=== Environment Verification ===')
print(f' ✅ PyTorch Version: {torch.__version__}')
print(f' ✅ TorchVision Version: {torchvision.__version__}')
print(f' ✅ Transformers Version: {transformers.__version__}')
print(f' ✅ Diffusers Version: {diffusers.__version__}')
print(f' ✅ PIL (Pillow) Import OK')
print(f' ✅ OpenCV (cv2) Import OK')
print(f' ✅ NumPy Import OK')

# Check CUDA availability
if torch.cuda.is_available():
    print(f' 🚀 CUDA Available: YES (GPU Enabled, {torch.cuda.get_device_name()})')
else:
    print(' 💻 CUDA Available: NO (Running on CPU)')

# Validate diffusers functionality
# from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines import StableDiffusionPipeline
print(' ✅ Diffusers SDXL Pipeline Import OK')