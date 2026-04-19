"""
llm_processor.py
================
Stage 1: Parse raw scene input → call local Qwen2.5-7B → output structured JSON
in the new panels-array format.

Output schema:
{
  "story_id": "01",
  "panels": [
    {
      "index": 1,
      "raw_text": "Lily makes breakfast in the kitchen.",
      "expanded_prompt": "...",
      "negative_prompt": "...",
      "reference_image": null
    },
    ...
  ]
}

Usage:
    python src/llm_processor.py \
        --input data/TaskA/01.txt \
        --output data/json_data/data01.json
"""

import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. 输入解析
# ============================================================

def parse_raw_input(raw_text: str) -> dict:
    """
    解析输入:
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
                {"index": 2, "text": "Lily looks out the window quietly."},   # 代词已替换
                {"index": 3, "text": "Lily sits down to eat with a book beside her."}
            ]
        }
    """
    blocks = re.split(r'\[SEP\]', raw_text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    scenes = []
    subject_name = None

    for i, block in enumerate(blocks):
        text = re.sub(r'\[SCENE-\d+\]\s*', '', block).strip()
        # 提取并记录主体名(来自第一帧的 <Name>)
        name_match = re.search(r'<(\w+)>', text)
        if name_match and subject_name is None:
            subject_name = name_match.group(1)
        # 去掉尖括号
        text = re.sub(r'<(\w+)>', r'\1', text)
        scenes.append({"index": i + 1, "text": text})

    if subject_name is None:
        subject_name = "Subject"

    # 把后续帧中的代词替换成主体名(she/he/it/they → SubjectName)
    scenes = _replace_pronouns(scenes, subject_name)

    return {"subject_name": subject_name, "scenes": scenes}


def _replace_pronouns(scenes: list, subject_name: str) -> list:
    """
    把第二帧及之后的代词替换为主体名。
    只处理句首大写代词(避免错改 'her book' 之类的物主代词为 SubjectName)。
    """
    pronouns = [
        r'\bShe\b', r'\bHe\b', r'\bIt\b', r'\bThey\b',
    ]
    pattern = re.compile('|'.join(pronouns))

    for s in scenes:
        if s['index'] == 1:
            continue
        s['text'] = pattern.sub(subject_name, s['text'], count=1)
    return scenes


# ============================================================
# 2. LLM Prompt 构造
# ============================================================

SYSTEM_PROMPT = r"""You are a master storyboard artist. Given a scene, output a JSON object containing a vivid, concise visual description.

## OUTPUT SCHEMA (Strict JSON only)
{
  "story_id": "__STORY_ID__",
  "panels": [
    {
      "index": 1,
      "raw_text": "<verbatim input>",
      "expanded_prompt": "<core visual description>",
      "negative_prompt": "<base negatives>",
      "reference_image": null
    }
  ]
}

## INSTRUCTIONS FOR expanded_prompt

1. **CONTENT ONLY**: Write ONLY the visual content of the scene. 
   - **DO NOT** include画质词 (e.g., 'high quality', 'sharp', '8k').
   - **DO NOT** include艺术风格 (e.g., 'storyboard', 'illustration', 'cel-shaded'). (These are added by the system later).
   
2. **STRUCTURE**: Use ONE single, dense, natural-language sentence.
   - Format: "[Character] doing [Action] in [Location], [Camera Angle], [Mood/Lighting]"
   - Example: "[Lily_001] pours cereal at the kitchen counter, medium shot, morning sunlight casting soft shadows, focused expression."

3. **CONTINUITY (Panels 2+)**:
   - Explicitly mention the character is wearing the "SAME OUTFIT" to lock appearance.
   - Mention persistent background elements briefly.

4. **BREVITY**: Be descriptive but ruthless. Remove adverbs and filler words.
"""

# 注意：我们不再让 LLM 写 "Clean storyboard..." 这一段，这段太占地方了。

def build_user_prompt(subject_name: str, scenes: list) -> str:
    """构造发送给 LLM 的 user prompt"""
    scene_text_lines = []
    for s in scenes:
        scene_text_lines.append(f"[SCENE-{s['index']}] {s['text']}")
        if s['index'] < len(scenes):
            scene_text_lines.append("[SEP]")
    scene_text = "\n".join(scene_text_lines)

    return (
        f"Character name: {subject_name}\n"
        f"Total panels: {len(scenes)}\n\n"
        f"Scene sequence:\n{scene_text}\n\n"
        f"Now output the JSON object following the schema above. "
        f"Remember:\n"
        f"  - raw_text MUST be the exact scene text (without [SCENE-N] tag).\n"
        f"  - reference_image MUST be null for every panel.\n"
        f"  - story_id MUST be the literal string \"__STORY_ID__\".\n"
        f"  - Output ONLY valid JSON, nothing else."
    )


# ============================================================
# 3. LLM 加载与推理
# ============================================================

def load_llm(llm_path: str):
    print(f"📦 Loading LLM from {llm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
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
                      max_new_tokens: int = 4096) -> str:
    """手动构造 Qwen2.5 ChatML 格式,避开 apply_chat_template 的兼容性问题"""
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ============================================================
# 4. JSON 解析与后处理
# ============================================================

def parse_llm_output(raw_output: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start != -1 and brace_end != -1:
        json_str = cleaned[brace_start:brace_end + 1]
    else:
        json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败: {e}")
        debug_path = "llm_raw_output_debug.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        print(f"   原始输出已保存到 {debug_path}")
        return None


def derive_story_id(input_path: str) -> str:
    """
    从输入文件路径推断 story_id。
    例如: data/TaskA/01.txt → "01"
         data/TaskA/test_05.txt → "05"
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    # 优先取连续数字段
    m = re.search(r'\d+', stem)
    return m.group(0) if m else stem


def post_process(data: dict, story_id: str, results_root: str,
                 raw_scenes: list) -> dict:
    """
    对 LLM 输出做后处理:
      1. 强制写入正确的 story_id
      2. 强制 raw_text 与解析结果一致(防止 LLM 改写)
      3. 注入 reference_image 路径(Panel 1: null;
         Panel N: results_root/{story_id}/panel_{N-1}.png)
    """
    data["story_id"] = story_id

    panels = data.get("panels", [])
    raw_map = {s["index"]: s["text"] for s in raw_scenes}

    for p in panels:
        idx = p.get("index")
        # 强制 raw_text 用解析后的原文(代词已替换)
        if idx in raw_map:
            p["raw_text"] = raw_map[idx]
        # 注入 reference_image
        if idx == 1:
            p["reference_image"] = None
        else:
            p["reference_image"] = os.path.join(
                results_root, story_id, f"panel_{idx - 1}.png"
            ).replace("\\", "/")

    return data


# ============================================================
# 5. 主流程
# ============================================================

def process(input_path: str, output_path: str, llm_path: str,
            results_root: str = "results"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"📖 读取输入: {input_path}")
    parsed = parse_raw_input(raw_text)
    story_id = derive_story_id(input_path)
    print(f"   story_id: {story_id}")
    print(f"   主体: {parsed['subject_name']}, 共 {len(parsed['scenes'])} 帧")
    for s in parsed['scenes']:
        print(f"     [SCENE-{s['index']}] {s['text']}")

    tokenizer, model = load_llm(llm_path)

    user_prompt = build_user_prompt(parsed["subject_name"], parsed["scenes"])

    print("\n🧠 Running LLM inference ...")
    raw_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT, user_prompt)
    print(f"✅ LLM 输出长度: {len(raw_output)} 字符")

    data = parse_llm_output(raw_output)
    if data is None:
        print("❌ 无法解析 LLM 输出,流程终止。")
        return None

    data = post_process(data, story_id, results_root, parsed["scenes"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 已保存: {output_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="LLM Prompt Processor (new panels-array schema)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="原始场景输入文件 (e.g., data/TaskA/01.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSON 路径 (e.g., data/json_data/data01.json)")
    parser.add_argument("--llm_path", type=str,
                        default="./models/llm/Qwen2.5-7B-Instruct",
                        help="本地 LLM 模型路径")
    parser.add_argument("--results_root", type=str, default="results",
                        help="图像输出根目录 (用于构造 reference_image 路径)")
    args = parser.parse_args()

    process(args.input, args.output, args.llm_path, args.results_root)


if __name__ == "__main__":
    main()
