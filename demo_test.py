import torch
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision
import os

# 忽略警告
torchvision.disable_beta_transforms_warning()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==========================================
# 1. LLM 部分 (保持不变)
# ==========================================
print("🤖 1. Testing LLM...")
LLM_PATH = "./models/llm/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

prompt = "你是一个专业的绘画大师，请用英文描述一幅包含以下元素的画：一只穿着宇航服的猫，在月球上喝咖啡，赛博朋克风格。只输出画面描述，不要输出其他文字。"
inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
outputs = llm_model.generate(**inputs, max_new_tokens=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
image_prompt = generated_text[len(prompt):].strip()

if not image_prompt or len(image_prompt) < 10:
    image_prompt = "A cute cat wearing an astronaut suit, drinking coffee on the moon, cyberpunk style"

print(f"✅ Prompt: {image_prompt}")

# ==========================================
# 2. SDXL + IP-Adapter 部分 (最终修正)
# ==========================================
print("\n🎨 2. Loading SDXL & IP-Adapter...")

SDXL_PATH = "models/sdxl/sd_xl_base_1.0.safetensors"
# IP_ADAPTER_PATH = "./models/ip_adapter"

# 加载 SDXL
pipe = StableDiffusionXLPipeline.from_single_file(
    SDXL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.to("cuda")

# 加载 IP-Adapter
# pipe.load_ip_adapter(
#     IP_ADAPTER_PATH, 
#     subfolder='', 
#     weight_name="ip-adapter_sdxl_vit-h.safetensors",
#     image_encoder_folder="image_encoder",
#     local_files_only=True,
#     torch_dtype=torch.float16
# )
# pipe.set_ip_adapter_scale(0.6)

# ==========================================
# 3. 关键修复：手动构建 added_cond_kwargs
# ==========================================
print("🚀 Generating...")

# 1. 维度修正：SDXL IP-Adapter (ViT-H) 需要 (Batch, Sequence_Length, Dim)
# Sequence_Length = 257 (1 cls token + 256 patch tokens)


image = pipe(
    prompt=image_prompt,
    negative_prompt="ugly, blurry, low quality",
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

image.save("demo_output.png")
print("✅ Success! Image saved to demo_output.png")
    