"""
llm_processor.py
================
Stage 1: Parse raw scene input → call local Qwen2.5-7B → output structured data.json

Usage:
    python src/llm_processor.py --input data/test_0/input.txt --output data/test_0/data.json
    python src/llm_processor.py --input data/test_0/input.txt --output data/test_0/data.json --llm_path ./models/llm/Qwen2.5-7B-Instruct
"""

import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. 输入解析：从原始文本提取主体名 + 各帧场景
# ============================================================

def parse_raw_input(raw_text: str) -> dict:
    """
    解析格式如下的输入：
        [SCENE-1] <Lily> makes breakfast in the kitchen.
        [SEP]
        [SCENE-2] She looks out the window quietly.
        [SEP]
        [SCENE-3] She sits down to eat with a book beside her.

    返回:
        {
            "subject_name": "Lily",
            "scenes": [
                {"index": 1, "text": "Lily makes breakfast in the kitchen."},
                {"index": 2, "text": "She looks out the window quietly."},
                {"index": 3, "text": "She sits down to eat with a book beside her."}
            ]
        }
    """
    # 按 [SEP] 分割
    blocks = re.split(r'\[SEP\]', raw_text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    scenes = []
    subject_name = None

    for i, block in enumerate(blocks):
        # 去掉 [SCENE-N] 标记
        text = re.sub(r'\[SCENE-\d+\]\s*', '', block).strip()
        # 从第一帧提取 <SubjectName>
        name_match = re.search(r'<(\w+)>', text)
        if name_match and subject_name is None:
            subject_name = name_match.group(1)
        # 去掉尖括号，保留名字本身
        text = re.sub(r'<(\w+)>', r'\1', text)
        scenes.append({"index": i + 1, "text": text})

    if subject_name is None:
        subject_name = "Subject"

    return {"subject_name": subject_name, "scenes": scenes}


# ============================================================
# 2. 构造 LLM Prompt（两阶段合一：推断 Profile + 填充 Panel）
# ============================================================

SYSTEM_PROMPT = r"""You are a visual story prompt engineer. Given a character name and a sequence of scene descriptions, you must output a single JSON object that can be used to generate a consistent sequence of storyboard illustrations.

## OUTPUT FORMAT (strict JSON, no extra text):

{
  "<SUBJECT_NAME>": {
    "SUBJECT_NAME": "<name>",
    "CATEGORY": "<human_female|human_male|animal_canine|animal_feline|object>",
    "BODY_DESC": "<concise build/age/size>",
    "HEAD_DESC": "<face or head features>",
    "COLOR_SCHEME": "<dominant colors of skin/fur/hair>",
    "OUTFIT_OR_MARKINGS": "<clothing or distinctive markings>",
    "SIGNATURE_FEATURE": "<ONE unique visual anchor visible in every panel>",
    "TEST_SET": {
      "Panel_1": {
        "CORE_ACTION": "<precise physical action with body positions>",
        "EXPRESSION": "<facial/animal expression + gaze direction>",
        "SHOT_TYPE": "<wide|medium|medium_close_up|close_up>",
        "CAMERA_ANGLE": "<eye_level|low_angle|high_angle|over_the_shoulder>",
        "LOCATION": "<specific place description>",
        "BG_ELEMENTS": "<3-4 background props, comma separated>",
        "MOOD": "<one evocative phrase>",
        "SPATIAL_LINK": "<ESTABLISHING for Panel 1, else relation to previous>",
        "CARRIED_PROPS": "<N/A for Panel 1, else carried-over objects>"
      },
      "Panel_2": { ... },
      ...
    }
  }
}

## RULES:
R1. If the name is a common human name, infer gender. If it's an animal word (Dog, Cat, Bird), use that species.
R2. Infer a plausible, specific everyday outfit/appearance even though the input doesn't specify one. Be concrete about colors and materials.
R3. SIGNATURE_FEATURE must be a VISIBLE, COLOR-SPECIFIC feature (e.g., "bright red scarf", "golden fur with white chest patch"), never abstract.
R4. CORE_ACTION must expand the brief scene text into a detailed physical description: specify limb positions, posture, movement direction.
R5. Maintain cross-panel consistency: same outfit, same signature feature, same environment style. Use SPATIAL_LINK and CARRIED_PROPS to enforce continuity.
R6. Choose SHOT_TYPE and CAMERA_ANGLE to best serve the narrative beat (establishing shots are wider, emotional moments are closer).
R7. Output ONLY the JSON object. No markdown fences, no explanation, no preamble.

## FALLBACK DEFAULTS (use if you cannot infer):
- human_female: shoulder-length dark brown hair, soft oval face, cream top + dark trousers
- human_male: short dark hair, lean build, grey jacket + dark slim trousers  
- animal_dog: medium golden retriever mix, amber coat, white chest patch
- animal_cat: grey tabby domestic shorthair, white paws, green eyes
"""


def build_user_prompt(subject_name: str, scenes: list) -> str:
    """构造发送给 LLM 的 user prompt"""
    scene_text = ""
    for s in scenes:
        scene_text += f"[SCENE-{s['index']}] {s['text']}\n"
        if s['index'] < len(scenes):
            scene_text += "[SEP]\n"

    return f"""Character name: {subject_name}
Total panels: {len(scenes)}

Scene sequence:
{scene_text.strip()}

Now output the JSON object following the schema above. Remember: output ONLY valid JSON, nothing else."""


# ============================================================
# 3. LLM 推理
# ============================================================

def load_llm(llm_path: str):
    """加载 Qwen2.5-7B-Instruct"""
    print(f"📦 Loading LLM from {llm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ LLM loaded.")
    return tokenizer, model


def run_llm_inference(tokenizer, model, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 2048) -> str:
    """调用 Qwen2.5 chat 模板进行推理"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen2.5-Instruct 支持 apply_chat_template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,       # 低温度 → 更稳定的结构化输出
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
        )

    # 只取生成部分
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# ============================================================
# 4. 后处理：清理 LLM 输出，解析为 JSON
# ============================================================

def parse_llm_output(raw_output: str) -> dict:
    """
    尝试从 LLM 输出中提取合法 JSON。
    处理常见问题：markdown 代码块包裹、多余文字等。
    """
    # 去掉 markdown 代码块
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # 尝试找到最外层的 { ... }
    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start != -1 and brace_end != -1:
        json_str = cleaned[brace_start:brace_end + 1]
    else:
        json_str = cleaned

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败: {e}")
        print(f"原始输出前500字符:\n{raw_output[:500]}")
        # 保存原始输出供调试
        with open("llm_raw_output_debug.txt", "w", encoding="utf-8") as f:
            f.write(raw_output)
        print("已保存原始输出到 llm_raw_output_debug.txt")
        return None


# ============================================================
# 5. 注入 reference_image 路径（供 pipeline_runner 使用）
# ============================================================

def inject_reference_paths(data: dict, output_dir: str) -> dict:
    """
    为 Panel 2+ 的每个 panel 添加 reference_image 字段，
    指向前一帧的输出路径。
    """
    for subject_name, subject_data in data.items():
        # 找到 TEST_SET 开头的 key
        test_key = None
        for k in subject_data:
            if k.startswith("TEST_SET"):
                test_key = k
                break
        if test_key is None:
            continue

        panels = subject_data[test_key]
        panel_keys = sorted(panels.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

        for i, pk in enumerate(panel_keys):
            if i == 0:
                panels[pk]["reference_image"] = None
            else:
                prev_pk = panel_keys[i - 1]
                prev_index = re.search(r'\d+', prev_pk).group()
                panels[pk]["reference_image"] = os.path.join(
                    output_dir, f"panel_{prev_index}.png"
                )

    return data


# ============================================================
# 6. 主函数
# ============================================================

def process(input_path: str, output_path: str, llm_path: str,
            results_dir: str = "results"):
    """完整处理流程"""
    # 读取输入
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"📖 读取输入: {input_path}")
    parsed = parse_raw_input(raw_text)
    print(f"   主体: {parsed['subject_name']}, 共 {len(parsed['scenes'])} 帧")

    # 加载 LLM
    tokenizer, model = load_llm(llm_path)

    # 构造 prompt
    user_prompt = build_user_prompt(parsed["subject_name"], parsed["scenes"])
    print(f"\n📝 User prompt:\n{user_prompt}\n")

    # 推理
    print("🧠 Running LLM inference ...")
    raw_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT, user_prompt)
    print(f"✅ LLM 输出长度: {len(raw_output)} 字符")

    # 解析 JSON
    data = parse_llm_output(raw_output)
    if data is None:
        print("❌ 无法解析 LLM 输出，流程终止。")
        return None

    # 注入 reference_image 路径
    # 输出目录基于输入路径推断
    test_name = os.path.basename(os.path.dirname(input_path))  # e.g., "test_0"
    result_dir = os.path.join(results_dir, test_name)
    data = inject_reference_paths(data, result_dir)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 已保存: {output_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="LLM Prompt Processor for Story Image Generation")
    parser.add_argument("--input", type=str, required=True,
                        help="原始场景输入文件路径 (e.g., ./data/TaskA/01.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSON 路径 (e.g., ./data/json_files/data01.json)")
    parser.add_argument("--llm_path", type=str, default="/data1/public/siqi.chen/DSAA2012_Final_Project/models/llm/Qwen2.5-7B-Instruct",
                        help="本地 LLM 模型路径")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="图像输出根目录")
    args = parser.parse_args()

    process(args.input, args.output, args.llm_path, args.results_dir)


if __name__ == "__main__":
    main()

