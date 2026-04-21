#!/usr/bin/env python3
# File: src/sdxl_generator.py
# Description: Optimized SDXL 1.0 + IP-Adapter + ControlNet with Single Pipeline.
import os
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, ControlNetModel
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import logging
from typing import Optional, Union

# 🔴 关键修复：启用可扩展内存段，对抗显存碎片（基于CUDA文档建议）
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDXLGenerator:
    def __init__(self, model_path: str = None, ip_adapter_path: str = None, device="cuda"):
        """
        初始化：单管线模式。IP-Adapter和ControlNet绝不预先加载。
        """
        self.device = device
        self.model_path = model_path or Path(__file__).parent.parent / "models" / "sdxl" / "sd_xl_base_1.0.safetensors"
        self.ip_adapter_path = ip_adapter_path or Path(__file__).parent.parent / "models" / "ip-adapter" / "sdxl_models" / "ip-adapter_sdxl.safetensors"
        
        print(f"🔄 加载SDXL模型 (单管线节能模式): {self.model_path}")
        
        # --- 1. 仅保留一条基础管线 ---
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path.as_posix(),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # --- 2. 状态标记（初始均为未加载）---
        self.ip_adapter_loaded = False
        self.controlnet = None
        self.controlnet_loaded = False
        self.default_control_weight = 0.37  # 平衡控制强度

        print(f"✅ SDXL 初始化完成。IP/ControlNet 处于休眠状态。")

    def _load_ip_adapter_if_needed(self):
        """仅在需要时加载IP-Adapter，节约显存"""
        if self.ip_adapter_loaded:
            return
        print(f"🌙 唤醒加载 IP-Adapter...")
        try:
            self.pipe.load_ip_adapter(
                self.ip_adapter_path.parent.parent.as_posix(),
                subfolder="sdxl_models",
                weight_name=self.ip_adapter_path.name, 
                local_files_only=True,
                torch_dtype=torch.float16
            )
            self.ip_adapter_loaded = True
        except Exception as e:
            logger.error(f"❌ IP-Adapter 加载失败: {e}")

    def _load_controlnet_if_needed(self):
        """仅在需要时加载ControlNet"""
        if self.controlnet_loaded:
            return
        logger.info("🌙 唤醒加载 ControlNet...")
        try:
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16,
                cache_dir="./models",
                variant="fp16"
            ).to(self.device)
            self.pipe.controlnet = self.controlnet
            self.controlnet_loaded = True
        except Exception as e:
            logger.error(f"❌ ControlNet 加载失败: {e}")

    @staticmethod
    def _create_clean_control_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """
        创建ControlNet控制图：保留背景线条，抹去人物细节。
        """
        # 1. 强模糊去噪，弱化人物纹理
        gray = image.convert('L')
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2.5))
        # 2. 提取边缘
        edge_img = blurred.filter(ImageFilter.FIND_EDGES)
        # 3. 二值化与去噪
        edge_array = np.array(edge_img)
        binary_array = np.where(edge_array > 100, 250, 50).astype(np.uint8)
        binary_img = Image.fromarray(binary_array)
        # 4. 形态学去噪 - 修复：使用奇数尺寸 (3)
        denoised = binary_img.filter(ImageFilter.MinFilter(size=3))
        smoothed = denoised.filter(ImageFilter.MaxFilter(size=3))
        return smoothed.resize((width, height), Image.Resampling.LANCZOS)

    def _prepare_controlnet_input(self, ctrl_ref_image, width: int, height: int) -> Optional[torch.Tensor]:
        """准备ControlNet输入张量"""
        try:
            if isinstance(ctrl_ref_image, str) and os.path.exists(ctrl_ref_image):
                source_img = Image.open(ctrl_ref_image).convert("RGB")
            else:
                source_img = ctrl_ref_image
            source_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # 生成控制图
            control_pil = self._create_clean_control_image(source_img, width, height)
            
            # 转换为Tensor [1, 3, H, W]
            control_array = np.array(control_pil.convert('L'))
            control_array = np.stack([control_array] * 3, axis=-1) # 3通道
            control_tensor = torch.from_numpy(control_array).float().div(255.0)
            control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            return control_tensor
        except Exception as e:
            logger.error(f"❌ 控制图生成失败: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        *,
        ip_ref_image: Optional[Union[str, Image.Image]] = None,
        ctrl_ref_image: Optional[Union[str, Image.Image]] = None,
        ctrl_weight: Optional[float] = None,
        width: int = 992,    # ✅ 992 ÷ 8 = 124 (完美对齐)
        height: int = 1024,  # ✅ 1024 ÷ 8 = 128
        num_inference_steps: int = 24,  # 稍减步数保显存
        guidance_scale: float = 8.6,
    ) -> bool:
        """
        显存安全版生成函数。
        """
        logger.info(f"🎨 生成: {prompt[:55]}...")
        
        # 初始化变量
        conditioning_scale = 0.0
        ip_adapter_scale_val = 0.0
        control_image_tensor = None
        final_ip_image = None
        
        # 🛡️ 负向提示
        base_neg_prompt = negative_prompt or ""
        base_neg_prompt += ", blurry, distorted, deformed, bad anatomy, extra limbs"

        try:
            # =============================================================
            # 🅰️ 情况 A：首帧生成 (无参考图，最省显存路径)
            # =============================================================
            if ip_ref_image is None and ctrl_ref_image is None:
                print("   🅰️ 轻量模式生成首帧...")
                # 关键：此时pipe未挂载IP和ControlNet，显存占用最低
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=base_neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            # =============================================================
            # 🅱️ 情况 B：后继帧 (动态挂载重量级组件)
            # =============================================================
            else:
                print("   🅱️ 增强模式生成...")
                
                # --- 1. ControlNet 背景控制 ---
                if ctrl_ref_image is not None:
                    self._load_controlnet_if_needed()
                    control_image_tensor = self._prepare_controlnet_input(ctrl_ref_image, width, height)
                    conditioning_scale = ctrl_weight or self.default_control_weight
                    if control_image_tensor is not None:
                        logger.info(f"🎛️  ControlNet 强度: {conditioning_scale:.2f}")
                    else:
                        logger.warning("⚠️ 控制图生成失败，跳过ControlNet")

                # --- 2. IP-Adapter 人物一致 ---
                if ip_ref_image is not None:
                    self._load_ip_adapter_if_needed()
                    if isinstance(ip_ref_image, str) and os.path.exists(ip_ref_image):
                        final_ip_image = load_image(ip_ref_image).convert("RGB")
                        final_ip_image = final_ip_image.resize((width, height), Image.Resampling.LANCZOS)
                    else:
                        final_ip_image = ip_ref_image
                    # 权重策略
                    ip_adapter_scale_val = 0.71 if control_image_tensor is not None else 0.79
                    logger.info(f"👤 IP-Adapter 强度: {ip_adapter_scale_val:.2f}")

                # --- 3. 叙事增强 ---
                enhanced_prompt = prompt
                if "looks out the window" in prompt.lower():
                    enhanced_prompt += ", side view, gazing out window"
                if "sits down" in prompt.lower():
                    enhanced_prompt += ", seated, book on table"

                # --- 4. 管线调用 ---
                call_kwargs = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": base_neg_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
                # 注入条件
                if control_image_tensor is not None:
                    call_kwargs["image"] = control_image_tensor
                    call_kwargs["controlnet_conditioning_scale"] = conditioning_scale
                if final_ip_image is not None:
                    call_kwargs["ip_adapter_image"] = final_ip_image
                    call_kwargs["ip_adapter_scale"] = ip_adapter_scale_val

                result = self.pipe(**call_kwargs).images[0]
                
                # 🔴 显存救急：立即释放大张量
                if control_image_tensor is not None:
                    del control_image_tensor

            # --- 5. 保存结果 ---
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            logger.info(f"💾 已保存: {output_path}")
            return True

        except torch.cuda.OutOfMemoryError:
            logger.error("💥 CUDA OOM！建议：1) 重启终端清空显存 2) 降低分辨率至896x832")
            torch.cuda.empty_cache()
            return False
        except Exception as e:
            logger.exception(f"❌ 生成失败: {str(e)[:120]}")
            return False
