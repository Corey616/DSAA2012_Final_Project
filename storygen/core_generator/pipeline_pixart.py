"""
PixArt-Σ Generation Pipeline.
Drop-in replacement for NarrativeGenerationPipeline using PixArtSigmaPipeline.

Architecture:
- Uses PixArt-Σ Transformer backbone (28 blocks) instead of SDXL UNet
- Same prompt compilation pipeline (model-agnostic)
- PixArtConsistentSelfAttentionProcessor for cross-frame consistency
- Batch generation with SCA window attention

Usage:
    from storygen.core_generator.pipeline_pixart import PixArtGenerationPipeline
    pipe = PixArtGenerationPipeline(config)
    images = pipe.generate_story(production_board)
"""

import torch
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from storygen.script_director.llm_parser import ProductionBoard, Panel
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.core_generator.attention.consistent_self_attn_pixart import PixArtConsistentSelfAttentionProcessor


class PixArtGenerationPipeline(NarrativeGenerationPipeline):
    """
    PixArt-Σ backbone variant of the story generation pipeline.

    Overrides:
    - base_pipe: loads PixArtSigmaPipeline instead of StableDiffusionXLPipeline
    - attn_processor: injects PixArtConsistentSelfAttentionProcessor
    - generate_story: adapted for PixArt API shape differences
    - No IP-Adapter face lock support (PixArt has different conditioning)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._pixart_model_name = config.get(
            "base_model",
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        )
        self._pixart_loaded = False

    @property
    def base_pipe(self):
        if self._base_pipe is None:
            from diffusers import PixArtSigmaPipeline
            from storygen.utils.mirror_config import get_models_cache_dir

            cache_dir = get_models_cache_dir()
            print(f"[PixArt] Loading PixArt-Σ Model: {self._pixart_model_name}")
            print(f"[PixArt] Using cache directory: {cache_dir}")

            self._base_pipe = PixArtSigmaPipeline.from_pretrained(
                self._pixart_model_name,
                torch_dtype=self.dtype,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            ).to(self.device)

            self._base_pipe.set_progress_bar_config(disable=True)
            print(f"[PixArt] Loaded {type(self._base_pipe.transformer).__name__} "
                  f"with {len(self._base_pipe.transformer.transformer_blocks)} blocks")
        return self._base_pipe

    @property
    def attn_processor(self):
        if self._attn_processor is None:
            consistency_strength = self.config.get("consistency_strength", 0.0)
            if consistency_strength > 0:
                window_size = self.config.get("sca_window_size", 1)
                start_ratio = self.config.get("sca_start_ratio", 0.25)
                print(f"[PixArt] Setting up SCA (strength={consistency_strength}, "
                      f"window={window_size}, start={start_ratio})...")

                _ = self.base_pipe
                num_blocks = len(self._base_pipe.transformer.transformer_blocks)
                processor_map = {}
                sca_count = 0

                for name, existing_processor in self._base_pipe.transformer.attn_processors.items():
                    if name.endswith("attn1.processor"):
                        block_idx = int(name.split(".")[1])
                        ratio = block_idx / max(num_blocks - 1, 1)
                        if ratio < 0.35:
                            layer_group = "early"
                        elif ratio < 0.65:
                            layer_group = "mid"
                        else:
                            layer_group = "late"

                        processor_map[name] = PixArtConsistentSelfAttentionProcessor(
                            consistency_strength=consistency_strength,
                            device=self.device,
                            layer_group=layer_group,
                            window_size=window_size,
                            apply_after_ratio=start_ratio,
                        )
                        sca_count += 1
                    else:
                        processor_map[name] = existing_processor

                self._attn_processor = processor_map
                self._base_pipe.transformer.set_attn_processor(self._attn_processor)
                print(f"[PixArt] SCA applied to {sca_count} self-attention layers "
                      f"({num_blocks} blocks, {len(processor_map)} total processors)")
            else:
                print("[PixArt] Using default attention (consistency disabled)")
                self._attn_processor = None
        return self._attn_processor

    def _set_sca_step_state(self, current_step: int, total_steps: int):
        if isinstance(self._attn_processor, dict):
            for proc in self._attn_processor.values():
                if isinstance(proc, PixArtConsistentSelfAttentionProcessor):
                    proc.set_step_state(current_step, total_steps)

    def _set_sca_story_context(self, story_state: Optional[Dict[str, Any]]):
        pair_weights = self._build_sca_pair_weights(story_state)
        if isinstance(self._attn_processor, dict):
            for proc in self._attn_processor.values():
                if isinstance(proc, PixArtConsistentSelfAttentionProcessor):
                    proc.set_story_context(pair_weights)

    @torch.inference_mode()
    def generate_story(
        self,
        production_board: ProductionBoard,
        seed: Optional[int] = None,
        return_portraits: bool = False,
    ) -> Tuple[List[Image.Image], Optional[Dict]]:
        if not self._initialized:
            _ = self.base_pipe
            _ = self.attn_processor

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[PixArt] Starting story: {production_board.story_id}")
        print(f"[PixArt] Total frames: {len(production_board.panels)}, Seed: {seed}")

        all_images = []
        gen_params = self.config.get("generation_params", {
            "num_steps": 20,
            "guidance_scale": 4.5,
        })

        height = self.config.get("height", 1024)
        width = self.config.get("width", 1024)
        num_frames = len(production_board.panels)

        _ = self.portrait_gen
        self._wardrobe_preferences = {}
        self._set_sca_story_context(production_board.story_state)

        prompts = []
        for i, panel in enumerate(production_board.panels):
            prompt = self._compose_prompt(
                panel=panel,
                global_style=production_board.global_style,
                characters=production_board.characters,
                panel_index=i,
                all_panels=production_board.panels,
                consistency_constraints=production_board.consistency_constraints,
                story_state=production_board.story_state,
                return_plan=False,
            )
            prompts.append(prompt)
            print(f"[Frame {i+1}/{num_frames}] Prompt: {prompt[:150]}...")

        negative_prompt = (
            "blurry, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, fused fingers, too many fingers, "
            "low quality, worst quality, jpeg artifacts, "
            "cartoon, anime style, illustration, painting, drawing"
        )

        total_steps = gen_params.get("num_steps", 20)
        self._set_sca_step_state(0, total_steps)

        print(f"\n[PixArt] Running batch inference ({num_frames} frames)...")
        try:
            output = self.base_pipe(
                prompt=prompts,
                negative_prompt=[negative_prompt] * num_frames,
                height=height,
                width=width,
                num_inference_steps=total_steps,
                guidance_scale=gen_params.get("guidance_scale", 4.5),
                generator=generator,
                output_type="pil",
            )

            for i, panel in enumerate(production_board.panels):
                img = output.images[i]
                all_images.append(img)
                print(f"[Frame {i+1}/{num_frames}] Completed")

        except Exception as e:
            print(f"[PixArt] Batch generation failed: {e}")
            import traceback
            traceback.print_exc()
            for i, panel in enumerate(production_board.panels):
                prompt = prompts[i] if i < len(prompts) else ""
                try:
                    frame_output = self.base_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=total_steps,
                        guidance_scale=gen_params.get("guidance_scale", 4.5),
                        generator=generator,
                        output_type="pil",
                    )
                    all_images.append(frame_output.images[0])
                    print(f"[Frame {i+1}/{num_frames}] Completed (sequential)")
                except Exception as e2:
                    print(f"[PixArt] Sequential frame {i+1} failed: {e2}")
                    placeholder = Image.new('RGB', (height, width), color=(128, 128, 128))
                    all_images.append(placeholder)

        print(f"\n[PixArt] Story generation complete! Generated {len(all_images)} frames\n")
        return all_images, None

    def save_story_images(
        self,
        images: List[Image.Image],
        story_id: str,
        panels: List[Panel],
        output_dir: str = "outputs/pixart_results",
    ) -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / f"{story_id}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, (img, panel) in enumerate(zip(images, panels)):
            filename = f"frame_{i+1:02d}_{panel.shot_type}.png"
            filepath = save_dir / filename
            img.save(filepath)
            saved_paths.append(str(filepath))
            print(f"[Save] Saved: {filepath.name}")

        storyboard = self._create_storyboard(images, panels)
        storyboard_path = save_dir / "storyboard.png"
        storyboard.save(storyboard_path)
        print(f"[Save] Storyboard saved: {storyboard_path.name}")

        return saved_paths
