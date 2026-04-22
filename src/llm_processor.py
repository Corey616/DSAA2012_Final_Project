#!/usr/bin/env python3
"""
llm_processor.py
================
Multi-turn Reasoning Architecture for Storyboard Generation.

Phase 1: Story Analysis - Analyze narrative structure, identify core elements, define panel boundaries
Phase 2: Visual Prompt Generation - Generate detailed prompts based on refined story descriptions

Key Design Principles:
- Explicit descriptions only (no "the same", no pronouns for visual elements)
- Multi-turn dialogue for iterative refinement
- Each turn preserves context from previous turns
"""

import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chinese network mirror configuration for HuggingFace models
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ============================================================
# Section 1: Input Parsing
# ============================================================

def parse_raw_input(raw_text: str) -> dict:
    """
    Parse raw input, preserve all pronouns and tags, no text replacement.
    """
    blocks = re.split(r'\[SEP\]', raw_text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    scenes = []
    default_subject = None

    for i, block in enumerate(blocks):
        text = re.sub(r'\[SCENE-\d+\]\s*', '', block).strip()
        
        if i == 0:
            name_match = re.search(r'<(\w+)>', text)
            if name_match:
                default_subject = name_match.group(1)
        
        scenes.append({"index": i + 1, "text": text})

    if default_subject is None:
        default_subject = "Subject"

    return {
        "subject_name": default_subject,
        "default_subject": default_subject,
        "scenes": scenes
    }

# ============================================================
# Section 2: Multi-Turn System Prompts
# ============================================================

# Character name to gender mapping (for reference)
NAME_GENDER_MAP = {
    "Ryan": "man", "Lily": "woman", "Jack": "man", "Sara": "woman",
    "Leo": "man", "Nina": "woman", "Max": "man", "Anna": "woman",
    "Tom": "man", "Lucy": "woman", "Ben": "man", "Emma": "woman",
    "David": "man", "Sophie": "woman", "Mike": "man", "Kate": "woman",
}

# Common hair colors for different names (fallback)
NAME_HAIR_MAP = {
    "Ryan": "short brown hair", "Lily": "blonde hair in ponytail",
    "Jack": "brown hair with beard", "Sara": "blonde wavy hair",
    "Leo": "brown hair", "Nina": "dark brown hair",
    "Max": "black short hair", "Anna": "long black hair",
    "Tom": "red short hair", "Lucy": "auburn hair",
    "Ben": "brown curly hair", "Emma": "brown long hair",
    "David": "black hair", "Sophie": "blonde hair",
    "Mike": "brown hair", "Kate": "black long hair",
}

SYSTEM_PROMPT_PHASE1 = r"""You are a story structure analyst for multi-panel storyboard generation. Your task is to analyze the narrative and break it down into coherent panels with detailed descriptions.

## Your Analysis Process

### Step 1: Character Identification
- List ALL characters appearing in the story
- For each character, determine: name, gender, hair style, outfit (top color, bottom color, shoes)
- Assign unique IDs: [CharacterName_001], [CharacterName_002], etc.

### Step 2: Core Scene Elements
- Identify objects that MUST appear across multiple panels (visual anchors)
- Examples: bus, kitchen counter, window, table, book, coffee cup

### Step 3: Panel-by-Panel Analysis
For EACH panel, provide:
1. **Scene Setting**: Where is this taking place?
2. **Characters Present**: Which characters appear in this panel?
3. **Action Description**: What is each character doing? (Use present continuous tense)
4. **Environmental Details**: What objects/background are visible?
5. **Temporal Context**: Time of day, lighting quality

### Step 4: Continuity Check
- List how each panel connects to the previous one
- Identify what objects remain in scene from previous panel
- Note any new objects introduced or old objects removed

## Output Format

Provide your analysis in this structured format:

```
STORY STRUCTURE ANALYSIS
========================

Characters:
- [CharacterID]: name, gender, hair style, outfit (top color + type, bottom color + type, shoe color + type)

Core Scene Elements (appear in multiple panels):
- [Element 1]: description
- [Element 2]: description

Panel 1 Analysis:
- Setting: [location description]
- Characters: [list of character IDs present]
- Actions: [detailed action description]
- Environment: [visible objects and background]
- Lighting: [lighting quality and direction]

Panel 2 Analysis:
- Setting: [location description]
- Characters: [list of character IDs present]
- Actions: [detailed action description]
- Environment: [visible objects - emphasize continuity from Panel 1]
- Lighting: [lighting quality and direction]
- Transition from Panel 1: [what connects this to Panel 1]

Panel 3 Analysis:
[...same structure...]
```

IMPORTANT:
- Be EXPLICIT about colors, shapes, and positions
- Do NOT use pronouns like "it", "they", "the same" - always describe explicitly
- Each panel's environment should reference objects from previous panels
"""

SYSTEM_PROMPT_PHASE2 = r"""You are a visual prompt engineer. Based on the story analysis provided, generate detailed image prompts for each panel.

## Critical Rules

### EXPLICIT Description Only
- NEVER use: "the same", "same as", "also wearing", "still has"
- ALWAYS use: Full explicit description of appearance in EVERY panel

### Required Description Elements Per Panel
1. Character Appearance (in EVERY panel, not abbreviated):
   - Character ID
   - Gender
   - Hair color and style (full description)
   - Top: color + garment type (e.g., "gray hoodie")
   - Bottom: color + garment type (e.g., "blue jeans")
   - Shoes: color + type (e.g., "black sneakers")

2. Action Description:
   - Verb in present continuous (-ing form)
   - Specific body movements (arms, legs, head position)
   - Facial expression hints

3. Scene Description:
   - Location setting
   - Key objects present (explicitly described with colors)
   - Background elements

4. Lighting:
   - Quality: morning light, afternoon light, evening light, dim light, soft light
   - Direction if notable

### Multi-Character Positioning
- Use LEFT and RIGHT explicitly
- Example: "[Character1] on the left, [Character2] on the right"
- NEVER assume spatial relationships - always state them

### Forbidden Terms
- "the same [clothing item]"
- "also wearing"
- "still wearing"
- "same as before"
- "like in previous panel"

### Negative Prompt Template
Use this exact format (modify character count based on story):
- Single character: "blurry, distorted, deformed, extra limbs, floating, wrong pose, stiff, static, wrong outfit color, wrong hairstyle, duplicate character"
- Multi-character: Add "character blending, merged, Siamese twins, swapped faces, three people when should be two, solo character when should be two"

## Output Format

Generate pure JSON:
{
  "story_id": "[number]",
  "panels": [
    {
      "index": 1,
      "raw_text": "[original scene text]",
      "expanded_prompt": "[EXPLICIT detailed prompt - no pronouns, no 'same', full description]",
      "negative_prompt": "[negative prompt]",
      "reference_image": null
    },
    {
      "index": 2,
      "raw_text": "[original scene text]",
      "expanded_prompt": "[EXPLICIT detailed prompt - include full character appearance again]",
      "negative_prompt": "[negative prompt]",
      "reference_image": "results/[id]/panel_1.png"
    }
  ]
}

IMPORTANT:
- expanded_prompt must contain FULL character description for EACH panel
- Do NOT abbreviate or use references to previous panels
- Keep prompt under 70 words but make it COMPLETE
"""

# ============================================================
# Section 3: Build User Prompts for Each Phase
# ============================================================

def build_phase1_user_prompt(scenes: list) -> str:
    """Build user prompt for Phase 1: Story Analysis"""
    scene_blocks = []
    for s in scenes:
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    # Detect characters
    all_text = " ".join([s['text'] for s in scenes])
    names = re.findall(r'<(\w+)>', all_text)
    unique_names = list(dict.fromkeys(names))
    
    return f"""Analyze the following story and break it down into panels:

{scenes_text}

Detected characters: {', '.join(unique_names)}
Number of panels to generate: {len(scenes)}

Please perform the full story structure analysis as instructed in your system prompt.
"""

def build_phase2_user_prompt(scenes: list, phase1_analysis: str) -> str:
    """Build user prompt for Phase 2: Visual Prompt Generation"""
    scene_blocks = []
    for s in scenes:
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    # Detect characters
    all_text = " ".join([s['text'] for s in scenes])
    names = re.findall(r'<(\w+)>', all_text)
    unique_names = list(dict.fromkeys(names))
    is_multi = len(unique_names) > 1
    
    character_count_hint = "MULTI-CHARACTER story" if is_multi else "SINGLE-CHARACTER story"
    
    return f"""Based on the following story analysis, generate explicit visual prompts for each panel:

## Original Story
{scenes_text}

## Story Analysis (from Phase 1)
{phase1_analysis}

Story type: {character_count_hint}
Number of panels: {len(scenes)}

## Instructions
1. Generate EXACTLY {len(scenes)} panels
2. For EACH panel, provide COMPLETE character descriptions (do NOT abbreviate)
3. NEVER use "the same", "also wearing", etc. - always write full explicit descriptions
4. Each prompt should describe: character appearance, action, scene, lighting
5. Output ONLY the JSON, no commentary
"""

def build_refinement_user_prompt(scenes: list, previous_prompts: list) -> str:
    """Build user prompt for Phase 3: Prompt Refinement (if needed)"""
    scene_blocks = []
    for s in scenes:
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    prompts_text = "\n\n".join([
        f"Panel {i+1}: {p}" for i, p in enumerate(previous_prompts)
    ])
    
    return f"""Review and refine the following visual prompts for better consistency:

## Original Story
{scenes_text}

## Current Prompts
{prompts_text}

## Review Checklist
1. Are all character appearances described explicitly in each panel?
2. Is the continuity between panels maintained (same objects, smooth transitions)?
3. Are there any pronouns like "the same", "also wearing" that should be replaced?
4. Is each panel's scene clearly defined with specific objects?

If any prompts need correction, provide the improved version. Output the complete refined JSON.
"""

# ============================================================
# Section 3.5: Backward Compatibility Layer (SYSTEM_PROMPT only)
# ============================================================

# Combined SYSTEM_PROMPT for single-stage mode (pipeline_runner compatibility)
SYSTEM_PROMPT = SYSTEM_PROMPT_PHASE1 + "\n\n" + SYSTEM_PROMPT_PHASE2

def build_user_prompt(default_subject: str, scenes: list) -> str:
    """
    Build user prompt for single-stage mode (backward compatibility).
    Uses multi-turn analysis internally but returns unified result.
    """
    scene_blocks = []
    for s in scenes:
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    # Detect characters
    all_text = " ".join([s['text'] for s in scenes])
    names = re.findall(r'<(\w+)>', all_text)
    unique_names = list(dict.fromkeys(names))
    is_multi = len(unique_names) > 1
    char_list = ", ".join([f"<{n}>" for n in unique_names])
    
    scene_type = "MULTI-CHARACTER" if is_multi else "SINGLE-CHARACTER"
    
    return f"""Analyze and generate prompts for the following story sequence:

{scenes_text}

Characters detected: {char_list}
Scene type: {scene_type}
Number of panels: {len(scenes)}

IMPORTANT:
1. First analyze the story structure (Characters, Core Elements, Panel-by-Panel breakdown)
2. Then generate explicit visual prompts for each panel
3. NEVER use "the same", "also wearing" - always describe character appearance explicitly in EACH panel
4. Include: character ID, gender, hair color+style, top color+type, bottom color+type, action, scene, lighting
5. Output pure JSON matching the schema, no commentary outside JSON.
"""

# ============================================================
# Section 4: LLM Loading and Inference
# ============================================================

def load_llm(llm_path: str):
    """Load LLM from local path or HuggingFace (via HF-Mirror)."""
    print(f"[LLM] Loading model: {llm_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        cache_dir="./models",
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="./models",
    )
    model.eval()
    print("[LLM] Model loaded successfully.")
    return tokenizer, model

def run_llm_inference(tokenizer, model, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 4096) -> str:
    """Qwen2.5 ChatML format inference"""
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
# Section 5: JSON Parsing
# ============================================================

def parse_json_output(raw_output: str) -> dict:
    """Extract JSON from LLM output"""
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()

    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start != -1 and brace_end != -1:
        json_str = cleaned[brace_start:brace_end + 1]
    else:
        json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parsing failed: {e}")
        with open("llm_debug.txt", "w", encoding="utf-8") as f:
            f.write(f"Raw output:\n{raw_output}\n\nCleaned:\n{cleaned}")
        print("[DEBUG] Raw output saved to llm_debug.txt")
        return None

# Alias for backward compatibility with pipeline_runner.py
parse_llm_output = parse_json_output

# ============================================================
# Section 6: Multi-Turn Processing Pipeline
# ============================================================

def process_multiturn(tokenizer, model, scenes: list, story_id: str, 
                      results_root: str, enable_refinement: bool = True) -> dict:
    """
    Multi-turn processing pipeline:
    Phase 1: Story Analysis
    Phase 2: Visual Prompt Generation
    Phase 3: Optional Refinement
    """
    print("\n" + "="*60)
    print("PHASE 1: STORY ANALYSIS")
    print("="*60)
    
    # Phase 1: Analyze story structure
    user_prompt_p1 = build_phase1_user_prompt(scenes)
    print(f"[INPUT] Analyzing {len(scenes)} scenes...")
    
    phase1_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT_PHASE1, user_prompt_p1)
    print(f"[PHASE 1 OUTPUT]\n{phase1_output[:500]}..." if len(phase1_output) > 500 else f"[PHASE 1 OUTPUT]\n{phase1_output}")
    
    print("\n" + "="*60)
    print("PHASE 2: VISUAL PROMPT GENERATION")
    print("="*60)
    
    # Phase 2: Generate visual prompts
    user_prompt_p2 = build_phase2_user_prompt(scenes, phase1_output)
    
    phase2_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT_PHASE2, user_prompt_p2)
    
    data = parse_json_output(phase2_output)
    
    if data is None:
        print("[ERROR] Phase 2 output parsing failed")
        return None
    
    # Validate panel count
    if len(data.get("panels", [])) != len(scenes):
        print(f"[WARN] Panel count mismatch: expected {len(scenes)}, got {len(data.get('panels', []))}")
        if enable_refinement:
            print("[INFO] Proceeding to Phase 3 for refinement...")
        else:
            # Try to fix by padding/truncating
            panels = data.get("panels", [])
            while len(panels) < len(scenes):
                panels.append(panels[-1].copy() if panels else {
                    "index": len(panels) + 1,
                    "raw_text": scenes[len(panels)]["text"] if len(panels) < len(scenes) else "",
                    "expanded_prompt": "placeholder",
                    "negative_prompt": "blurry, distorted, deformed",
                    "reference_image": None
                })
            data["panels"] = panels[:len(scenes)]
    
    if enable_refinement:
        print("\n" + "="*60)
        print("PHASE 3: PROMPT REFINEMENT")
        print("="*60)
        
        # Phase 3: Refine prompts
        prompts_only = [p.get("expanded_prompt", "") for p in data.get("panels", [])]
        user_prompt_p3 = build_refinement_user_prompt(scenes, prompts_only)
        
        phase3_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT_PHASE2, user_prompt_p3)
        
        refined_data = parse_json_output(phase3_output)
        if refined_data is not None:
            data = refined_data
            print("[PHASE 3] Refinement complete")
        else:
            print("[PHASE 3] Refinement failed, using Phase 2 output")
    
    return data

# ============================================================
# Section 7: Post-processing
# ============================================================

def derive_story_id(input_path: str) -> str:
    """Extract numeric ID from filename"""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    m = re.search(r'\d+', stem)
    return m.group(0) if m else stem

def post_process(data: dict, story_id: str, results_root: str, raw_scenes: list) -> dict:
    """Post-processing: Inject IDs, paths, and force alignment with original Raw Text."""
    data["story_id"] = story_id
    panels = data.get("panels", [])
    raw_text_map = {s["index"]: s["text"] for s in raw_scenes}

    for p in panels:
        idx = p["index"]
        if idx in raw_text_map:
            p["raw_text"] = raw_text_map[idx]
        
        if idx == 1:
            p["reference_image"] = None
        else:
            ref_path = os.path.join(results_root, story_id, f"panel_{idx-1}.png")
            p["reference_image"] = ref_path.replace("\\", "/")
            
    return data

# ============================================================
# Section 8: Main Process
# ============================================================

def process(input_path: str, output_path: str, llm_path: str, 
            results_root: str = "results", enable_refinement: bool = True):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[INPUT] Parsing: {input_path}")
    parsed = parse_raw_input(raw_text)
    story_id = derive_story_id(input_path)
    
    print(f"[INPUT] StoryID: {story_id}, Subject: {parsed['subject_name']}, Frames: {len(parsed['scenes'])}")
    for s in parsed["scenes"]:
        print(f"  [SCENE-{s['index']}] {s['text']}")

    # Load LLM
    tokenizer, model = load_llm(llm_path)
    
    # Multi-turn processing
    print(f"\n[PROCESSING] Running multi-turn pipeline (refinement={'enabled' if enable_refinement else 'disabled'})...")
    
    data = process_multiturn(
        tokenizer, model, 
        parsed["scenes"], 
        story_id, 
        results_root,
        enable_refinement
    )
    
    if data is None:
        print("[ERROR] Processing failed, aborting")
        return None

    # Post-processing injection
    data = post_process(data, story_id, results_root, parsed["scenes"])
    
    # Save JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OUTPUT] JSON saved: {output_path}")
    
    # Print generated prompts for review
    print("\n" + "="*60)
    print("GENERATED PROMPTS")
    print("="*60)
    for p in data.get("panels", []):
        print(f"\n[Panel {p['index']}]")
        print(f"Raw: {p['raw_text']}")
        print(f"Prompt: {p['expanded_prompt'][:200]}..." if len(p.get('expanded_prompt', '')) > 200 else f"Prompt: {p.get('expanded_prompt', '')}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Multi-Turn LLM Processor for Storyboard Generation")
    parser.add_argument("--input", type=str, required=True, help="Input .txt file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model repo_id or local path")
    parser.add_argument("--results_root", type=str, default="results", help="Image output root directory")
    parser.add_argument("--no_refinement", action="store_true", help="Disable Phase 3 refinement")
    args = parser.parse_args()
    
    process(args.input, args.output, args.llm_path, args.results_root, 
            enable_refinement=not args.no_refinement)

if __name__ == "__main__":
    main()
