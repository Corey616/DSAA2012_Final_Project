"""
Local LLM Script Parser - Using Qwen2.5-7B-Instruct via Transformers
Fully portable solution that doesn't require external services

Advantages over Ollama API approach:
- ✅ Pure Python (no external service dependencies)
- ✅ Works in any standard Python environment
- ✅ Easier to deploy in academic evaluation settings
- ✅ Full control over memory and inference
- ✅ Can be packaged with the project
"""

import os
import torch
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys


# Setup China mirror FIRST before any model downloads
from src.utils.mirror_config import setup_china_mirrors, verify_model_integrity
setup_china_mirrors()

from src.script_director.llm_parser import (
    Character, Panel, ProductionBoard,
    LLMScriptParser
)


class LocalQwenParser(LLMScriptParser):
    """
    Local Qwen2.5-7B Parser using HuggingFace Transformers

    Loads Qwen2.5-7B-Instruct locally and uses it for script parsing.
    Fully compatible with the existing pipeline architecture.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 2048,
        use_cache: bool = True
    ):
        super().__init__(llm_backend="local_transformers", model_name=model_name_or_path)

        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache

        self._tokenizer = None
        self._model = None

        print("[LocalQwen] Initializing local Qwen2.5-7B parser...")
        print(f"[LocalQwen]   Model: {model_name_or_path}")
        print(f"[LocalQwen]   Device: {device_map}")
        print(f"[LocalQwen]   Precision: {'FP16' if torch_dtype == torch.float16 else 'FP32'}")

    def _get_local_model_path(self) -> Optional[str]:
        """
        Get the local cache path for the model.
        Returns the snapshot path if the model is cached, None otherwise.
        """
        from src.utils.mirror_config import get_models_cache_dir
        cache_dir = get_models_cache_dir()
        model_cache_path = cache_dir / f"models--{self.model_name_or_path.replace('/', '--')}"
        snapshots_dir = model_cache_path / "snapshots"
        
        if snapshots_dir.exists():
            snapshot_dirs = list(snapshots_dir.iterdir())
            if len(snapshot_dirs) > 0:
                return str(snapshot_dirs[0])
        return None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            from src.utils.mirror_config import get_models_cache_dir

            cache_dir = get_models_cache_dir()
            print(f"[LocalQwen] Loading tokenizer... (cache: {cache_dir})")
            
            # Try to use local path first
            local_path = self._get_local_model_path()
            model_path = local_path if local_path else self.model_name_or_path
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                local_files_only=(local_path is not None)
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            print("[LocalQwen] Tokenizer loaded")
        return self._tokenizer

    @property
    def model(self):
        """Lazy load model with cache-first strategy"""
        if self._model is None:
            from transformers import AutoModelForCausalLM
            from src.utils.mirror_config import get_models_cache_dir

            cache_dir = get_models_cache_dir()
            print(f"[LocalQwen] Loading model (this may take a moment)...")
            print(f"[LocalQwen] Using cache directory: {cache_dir}")

            # Check cache integrity first
            is_complete = verify_model_integrity(self.model_name_or_path, cache_dir)

            # Try to use local path first
            local_path = self._get_local_model_path()
            model_path = local_path if local_path else self.model_name_or_path

            if is_complete:
                print("[LocalQwen] ✓ Using local cache (skip network verification)")
            else:
                print("[LocalQwen] ⚠ Cache incomplete/missing, will download from mirror...")

            try:
                # Try loading with 4-bit quantization first (saves ~60% memory)
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )

                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=(local_path is not None)
                    )
                    print("[LocalQwen] Model loaded (4-bit quantized)")

                except ImportError:
                    print("[LocalQwen] BitsAndBytes not available, using FP16...")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=(local_path is not None)
                    )
                    print("[LocalQwen] Model loaded (FP16)")

                self._model.eval()

            except Exception as e:
                print(f"[LocalQwen] Failed to load model: {e}")
                raise RuntimeError(f"Cannot load model {self.model_name_or_path}: {e}")

        return self._model

    def _initialize_client(self):
        """Override: No client needed for local model"""
        self.client = "local_model"

    def _format_chat_prompt(self, user_content: str) -> str:
        """Format prompt using Qwen's chat template"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            return f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    def call_llm_for_analysis(self, parsed_script: Dict) -> str:
        """Call local Qwen for deep script analysis — LLM-only, no fallback."""
        try:
            _ = self.tokenizer
            _ = self.model
        except Exception as e:
            raise RuntimeError(
                f"[LocalQwen] Failed to load model/tokenizer: {e}. "
                "LLM-only parsing enforced. Ensure model is cached and GPU is available."
            ) from e

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "[LocalQwen] Model/tokenizer not loaded. "
                "LLM-only parsing enforced."
            )

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            script_text=parsed_script["raw_text"]
        )

        formatted_prompt = self._format_chat_prompt(user_prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self._model.device)

        print(f"[LocalQwen] Generating analysis (max_tokens={self.max_new_tokens})...")

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache
                )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            print(f"[LocalQwen] Analysis generated ({len(response_text)} chars)")

            return self._clean_response(response_text)

        except torch.cuda.OutOfMemoryError:
            print("[LocalQwen] CUDA out of memory! Trying reduced output...")
            try:
                self._model = self._model.to('cpu')
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs.to('cpu'),
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return self._clean_response(response)
            except Exception:
                raise RuntimeError(
                    "[LocalQwen] CUDA OOM recovery failed. "
                    "LLM-only parsing enforced. Free GPU memory and retry."
                )

        except Exception as e:
            raise RuntimeError(
                f"[LocalQwen] Generation error: {e}. "
                "LLM-only parsing enforced."
            ) from e

    def _clean_response(self, text: str) -> str:
        """Clean model response text"""
        text = text.strip()

        if text.startswith("```"):
            first_newline = text.find('\n')
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            elif "```\n" in text:
                text = text.split("```\n")[0]
            text = text.strip()

        if not text.startswith('{'):
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        return text

    def _infer_gender(self, name: str) -> str:
        """
        Infer gender from name using multiple strategies:
        1. Role-based names (Girl, Boy, Woman, Man)
        2. Common name patterns
        3. Common name endings
        """
        name_lower = name.lower()
        
        # === PRIORITY 1: Role-based gender indicators ===
        # These should always be checked first as they are explicit gender markers
        explicit_female_markers = {
            'girl', 'woman', 'female', 'lady', 'madam', 'miss', 'mrs', 'ms',
            'she', 'her', 'mom', 'mother', 'sister', 'daughter', 'queen', 'princess'
        }
        explicit_male_markers = {
            'boy', 'man', 'male', 'sir', 'mr', 'dad', 'father', 'brother', 'son',
            'king', 'prince', 'guy', 'chap', 'bloke'
        }
        
        # Check if name contains any explicit gender marker
        for marker in explicit_female_markers:
            if marker in name_lower:
                return "female"
        for marker in explicit_male_markers:
            if marker in name_lower:
                return "male"
        
        # === PRIORITY 2: Common female names ===
        female_names = {
            'lily', 'sara', 'sarah', 'emma', 'anna', 'mary', 'jane', 'alice', 'eve', 
            'grace', 'helen', 'iris', 'julia', 'kate', 'mia', 'nina', 'rose',
            'ivy', 'jade', 'amy', 'ava', 'bella', 'chloe', 'daisy', 'ella', 'fiona',
            'gina', 'hannah', 'isabel', 'jenny', 'karen', 'laura', 'megan', 'nancy', 'olivia',
            'peggy', 'quinn', 'rachel', 'sally', 'tina', 'uma', 'violet', 'wendy', 'xena',
            'yuki', 'zara', 'lucy', 'luna', 'linda', 'lisa', 'maggie', 'olga',
            'pam', 'rita', 'sandy', 'vicky', 'wanda', 'yvonne',
            'zoe', 'alexis', 'aubrey', 'bailey', 'camila', 'danielle', 'evelyn',
            'ginny', 'hayley', 'ingrid', 'joan', 'kayla', 'lena', 'monica', 'nora', 'patty',
            'sophia', 'olivia', 'amelia', 'isabella', 'mia', 'charlotte', 'harper', 'evelyn',
            'abigail', 'ella', 'scarlett', 'grace', 'layla', '秋', '夏', '春', '冬'  # Chinese seasons as female names
        }
        
        # === PRIORITY 3: Common male names ===
        male_names = {
            'jack', 'john', 'james', 'bob', 'mike', 'tom', 'dan', 'alex', 'charlie', 'david',
            'eric', 'fred', 'george', 'henry', 'ivan', 'kevin', 'larry', 'mark',
            'nick', 'oscar', 'peter', 'ray', 'steve', 'tim', 'victor', 'will', 'xavier',
            'yuri', 'zack', 'adam', 'ben', 'chris', 'dave', 'evan', 'frank', 'greg', 'hal',
            'ian', 'jeff', 'kyle', 'leo', 'matt', 'noah', 'owen', 'paul', 'ryan', 'sam',
            'todd', 'vince', 'walt', 'xander', 'yale', 'zach', 'arthur', 'bruce',
            'carl', 'dean', 'edgar', 'floyd', 'gavin', 'howard', 'isaac', 'jason', 'kurt',
            'louis', 'mason', 'neal', 'otto', 'phil', 'quincy', 'rick', 'scott', 'trevor',
            'milo', 'lucas', 'liam', ' ethan', 'mason', 'logan', 'jackson', 'sebastian',
            'aiden', 'matthew', 'samuel', 'david', 'joseph', 'carter', 'owen', 'wyatt',
            'john', 'dent', 'moriarty'  # Some character names
        }
        
        if name_lower in female_names:
            return "female"
        elif name_lower in male_names:
            return "male"
        else:
            # === PRIORITY 4: Common endings ===
            # Names ending in common vowel patterns
            if len(name) >= 3:
                if name_lower.endswith(('a', 'e', 'i', 'y', 'ie', 'ey', 'ine', 'ette')):
                    return "female"
                if name_lower.endswith(('n', 's', 'r', 't', 'o', 'en', 'son')):
                    # But exclude some common male endings
                    if not name_lower.endswith(('son', 'ton', 'man')):
                        return "male"
            return "male"  # Default to male if uncertain
    
    def _detect_age_category(self, name: str, script_context: str) -> str:
        """
        Detect if a character is likely a child, teen, adult, or elderly.
        Check both the name and the surrounding script context.
        """
        name_lower = name.lower()
        context_lower = script_context.lower() if script_context else ""
        
        # === EXPLICIT AGE MARKERS IN NAME ===
        explicit_child_markers = {'kid', 'child', 'children', 'boy', 'girl', 'baby', 'toddler', 'little'}
        explicit_adult_markers = {'man', 'woman', 'adult', 'senior', 'elder', 'mr', 'mrs', 'ms', 'dr'}
        
        for marker in explicit_child_markers:
            if marker in name_lower:
                return "child"
        for marker in explicit_adult_markers:
            if marker in name_lower:
                return "adult"
        
        # === SCRIPT CONTEXT INDICATORS ===
        # Look for age-related words near the character name in script
        child_indicators = [
            'child', 'kid', 'kids', 'children', 'boy', 'girl', 'boys', 'girls',
            'little', 'young', 'small', 'tall', 'tiny', 'toddler', 'infant', 'baby',
            'plays with toys', 'toy', 'toys', 'crib', 'stroller', 'children\'s',
            'school', 'homework', 'schoolbag', 'backpack', 'nap', 'nap time'
        ]
        adult_indicators = [
            'adult', 'grown', 'grown-up', 'office', 'meeting', 'work', 'job',
            'commute', 'drives', 'driving', 'professional', 'career'
        ]
        elderly_indicators = [
            'elderly', 'senior', 'old man', 'old woman', 'grandma', 'grandfather',
            'grandparent', 'retired'
        ]
        
        # Count indicators
        child_score = sum(1 for ind in child_indicators if ind in context_lower)
        adult_score = sum(1 for ind in adult_indicators if ind in context_lower)
        elderly_score = sum(1 for ind in elderly_indicators if ind in context_lower)
        
        # Check specific phrases that indicate children
        child_phrases = ['plays on the floor', 'sits on the floor', 'with toys', 'on a bridge',
                        'runs in the rain', 'watches the rain', 'look out the window',
                        'reads a book', 'with a book']
        
        if any(phrase in context_lower for phrase in child_phrases):
            # These activities are common for both children and adults
            # but combined with certain names/context can indicate children
            pass
        
        # Decision based on scores
        if child_score >= 2 or (child_score >= 1 and adult_score == 0):
            return "child"
        elif elderly_score >= 1:
            return "elderly"
        elif adult_score >= 1:
            return "adult"
        
        # Default based on scene context
        if any(word in context_lower for word in ['kitchen', 'breakfast', 'home']):
            return "adult"  # Typical morning routine
        elif any(word in context_lower for word in ['bridge', 'rain', 'scenery']):
            return "adult"  # More contemplative scenes
        
        return "adult"  # Default to adult
    
    def _extract_story_context(self, scenes: List, characters: List[str]) -> Dict:
        """
        CRITICAL FIX: Extract story-level context for timeline and key objects tracking.
        This ensures breakfast stories stay morning, key objects appear consistently, etc.
        """
        story_context = {
            'is_breakfast_story': False,
            'is_indoor_story': False,
            'key_objects': [],
            'total_characters': len(characters)
        }
        
        if not scenes:
            return story_context
        
        all_scene_text = " ".join(s.get("content", "") for s in scenes).lower()
        
        # Detect breakfast story (breakfast → eating)
        if any(word in all_scene_text for word in ['breakfast', 'breakfast in the kitchen']) and \
           any(word in all_scene_text for word in ['eat', 'eating', 'dining', 'sits down']):
            story_context['is_breakfast_story'] = True
            story_context['key_objects'].append("breakfast food on table")
        
        # Detect indoor story
        if any(word in all_scene_text for word in ['kitchen', 'home', 'cafe', 'office', 'room']):
            story_context['is_indoor_story'] = True
        
        # Extract key objects from entire story
        if 'breakfast' in all_scene_text or 'food' in all_scene_text:
            story_context['key_objects'].append("breakfast")
        if 'book' in all_scene_text or 'reading' in all_scene_text:
            story_context['key_objects'].append("book")
        if 'coffee' in all_scene_text:
            story_context['key_objects'].append("coffee cup")
        
        return story_context
    
    def _infer_story_time(self, scene_lower: str) -> str:
        """
        CRITICAL FIX: Infer the primary time of day for the story.
        Breakfast stories = morning, etc.
        """
        if any(word in scene_lower for word in ['breakfast', 'morning']):
            return "morning"
        elif any(word in scene_lower for word in ['lunch', 'midday', 'noon']):
            return "midday"
        elif any(word in scene_lower for word in ['dinner', 'evening', 'sunset']):
            return "evening"
        elif any(word in scene_lower for word in ['night', 'dark', 'midnight']):
            return "night"
        return "morning"  # Default to morning for home/kitchen scenes
    
    def _infer_primary_setting(self, scene_lower: str) -> str:
        """
        CRITICAL FIX: Infer the primary setting for the story.
        This prevents drifting from kitchen to "dimly lit dining room".
        """
        if 'kitchen' in scene_lower:
            return "cozy kitchen interior, home environment"
        elif 'cafe' in scene_lower or 'coffee' in scene_lower:
            return "cozy cafe interior"
        elif 'office' in scene_lower:
            return "modern office interior"
        # CRITICAL: Don't hardcode bus/train - use actual story context
        elif any(word in scene_lower for word in ['bus stop', 'bus station', 'bus stop']):
            return "bus stop, outdoor"
        elif any(word in scene_lower for word in ['train station', 'railway']):
            return "train station, outdoor"
        return None
    
    def _get_distinctive_features(self, name: str, gender: str, char_index: int, 
                                   age_category: str = "adult") -> str:
        """Generate distinctive visual features based on gender, age, and position"""
        import random
        random.seed(hash(name) % 2**32)
        
        # Default values
        hair_colors = ["black hair", "brown hair", "blonde hair", "dark brown hair"]
        hair_styles = ["short hair", "medium-length hair", "messy hair", "wavy hair"]
        eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
        builds = ["slim", "average", "athletic"]
        skin_tones = ["fair skin", "medium skin tone", "olive skin"]
        
        # Age-appropriate features
        if age_category == "child":
            if gender == "female":
                hair_colors = ["black hair", "brown hair", "blonde hair", "dark brown hair", "red hair"]
                hair_styles = ["long hair", "medium-length hair", "short hair", "wavy hair", "ponytail", "twin braids"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["slim", "average", "small build"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tan skin"]
            else:
                hair_colors = ["black hair", "brown hair", "dark brown hair", "blonde hair", "red hair"]
                hair_styles = ["short hair", "medium-length hair", "messy hair", "curly hair", "side part"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["slim", "average", "small build"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tan skin"]
        elif age_category == "elderly":
            if gender == "female":
                hair_colors = ["gray hair", "white hair", "silver hair", "light brown hair"]
                hair_styles = ["short hair", "medium-length hair", "curly hair", "styled hair"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["slim", "average", "slight build"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tanned skin"]
            else:
                hair_colors = ["gray hair", "white hair", "silver hair", "light brown hair"]
                hair_styles = ["short hair", "medium-length hair", "receding hairline", "bald"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["slim", "average", "slight build"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tanned skin"]
        else:  # adult
            if gender == "female":
                hair_colors = ["black hair", "brown hair", "blonde hair", "auburn hair", "dark brown hair"]
                hair_styles = ["long hair", "medium-length hair", "short hair", "wavy hair", "straight hair", "ponytail"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["slim", "average", "athletic"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin"]
            else:
                hair_colors = ["black hair", "brown hair", "dark brown hair", "light brown hair", "gray hair"]
                hair_styles = ["short hair", "medium-length hair", "buzz cut", "messy hair", "side part hair"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes"]
                builds = ["average", "athletic", "slim"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tan skin"]
        
        features = [
            random.choice(hair_colors),
            random.choice(hair_styles),
            random.choice(eye_colors),
            f"{random.choice(builds)} build",
            random.choice(skin_tones)
        ]
        return ", ".join(features[:3])  # Limit to 3 features for brevity
    
    def _analyze_scene_context(self, scene_content: str, scene_lower: str, characters: List[str],
                                char_appearances: Dict, is_first_scene: bool, is_last_scene: bool,
                                prev_setting: Optional[str] = None, 
                                story_context: Optional[Dict] = None) -> tuple:
        """
        Analyze scene context to determine proper setting, lighting, and narrative elements.
        This is the KEY METHOD for fixing narrative logic errors.
        """
        if story_context is None:
            story_context = {'is_indoor_story': False, 'has_transitioned': False, 'detected_vehicles': []}
        
        setting = "indoor scene"
        lighting = "natural"
        time_of_day = "daytime"
        mood_desc = "neutral atmosphere"
        action_desc = scene_content
        
        # === NARRATIVE FLOW ANALYSIS ===
        # Track if we're transitioning (e.g., going to bus, boarding bus, inside bus)
        # ORDER MATTERS: Check more specific patterns first
        
        # Check for "at the door" - threshold scene (NOT inside yet)
        is_at_door = any(word in scene_lower for word in ['at the door', 'pauses at', 'stops at', 'doorway', 'at doorway'])
        
        # Check for "gets inside" or "is inside" - actually inside
        is_boarding = any(word in scene_lower for word in ['gets inside', 'get inside', 'is inside', 'boards', 'board'])
        is_boarding = is_boarding or (any(word in scene_lower for word in ['get on', 'get in', 'enter', 'step on', 'climb']))
        
        is_exiting = any(word in scene_lower for word in ['get off', 'exit', 'leave', 'depart', 'get out'])
        
        # Check for approach/transit scene
        is_transition = any(word in scene_lower for word in ['toward', 'towards', 'heading', 'approaching', 'going to', 'walk to', 'head to'])
        
        # Check for "sit" - but only relevant if we're already inside
        has_sit = 'sit' in scene_lower or 'seated' in scene_lower
        
        # Get detected vehicles from story context
        detected_vehicles = story_context.get('detected_vehicles', [])
        has_transitioned = story_context.get('has_transitioned', False)
        
        # CRITICAL FIX: Only use vehicle context if explicitly detected in THIS story
        # Don't assume every story is about buses/trains!
        is_vehicle_story = len(detected_vehicles) > 0 and any(v in scene_lower for v in detected_vehicles)
        
        # Determine setting based on actual story content, not hardcoded logic
        if is_vehicle_story:
            # Vehicle story - but be flexible about which vehicle
            # Use actual vehicle name, not generic "bus/train"
            vehicle_name = next((v for v in detected_vehicles if v in scene_lower), detected_vehicles[0] if detected_vehicles else None)
            
            if is_at_door and vehicle_name:
                setting = f"near a {vehicle_name}, outdoor environment"
                lighting = "natural daylight"
                mood_desc = "outdoor scene"
            elif is_boarding or has_transitioned:
                if vehicle_name in ['bus', 'train']:
                    setting = f"inside {vehicle_name}, interior view"
                    lighting = "natural window light"
                    mood_desc = "indoor vehicle scene"
                else:
                    setting = "outdoor environment"
                    lighting = "natural daylight"
                    mood_desc = "outdoor scene"
            elif is_exiting:
                setting = "outdoor environment, exit area"
                lighting = "natural daylight"
                mood_desc = "outdoor arrival"
            elif is_transition:
                setting = "outdoor environment"
                lighting = "natural daylight"
                mood_desc = "outdoor scene"
            else:
                setting = "outdoor environment"
                lighting = "natural daylight"
                mood_desc = "outdoor scene"
        else:
            # Non-vehicle stories - existing logic
            
            # === OUTDOOR SCENES - Check these FIRST before indoor ===
            if any(word in scene_lower for word in ['bridge', 'over bridge', 'on a bridge']):
                setting = "outdoor bridge, city skyline, urban environment"
                lighting = "natural daylight"
                mood_desc = "contemplative outdoor atmosphere"
            
            elif any(word in scene_lower for word in ['rain', 'raining', 'rainy', 'storm']):
                setting = "outdoor, rain falling, urban street"
                lighting = "overcast, soft natural light"
                mood_desc = "rainy atmosphere"
                if 'roof' in scene_lower or 'cover' in scene_lower:
                    setting = "outdoor, standing under roof cover, rain visible"
                    mood_desc = "seeking shelter from rain"
            
            elif any(word in scene_lower for word in ['drive', 'driving', 'car', 'road', 'along a road']):
                setting = "outdoor, car driving, scenic road"
                lighting = "natural daylight"
                mood_desc = "road trip atmosphere"
            
            elif any(word in scene_lower for word in ['scenery', 'view', 'landscape']) and 'outside' in scene_lower:
                setting = "outdoor, scenic view, landscape"
                lighting = "natural daylight"
                mood_desc = "scenic outdoor atmosphere"
            
            elif any(word in scene_lower for word in ['park', 'garden']):
                setting = "outdoor park, trees, green nature"
                lighting = "natural daylight"
                mood_desc = "peaceful outdoor atmosphere"
            
            elif any(word in scene_lower for word in ['walk', 'walking', 'outdoor', 'outside']) and not any(word in scene_lower for word in ['indoor', 'inside', 'room', 'home', 'cafe', 'office']):
                setting = "outdoor, urban street"
                lighting = "natural daylight"
                mood_desc = "outdoor walking scene"
            
            elif any(word in scene_lower for word in ['street', 'road', 'downtown', 'urban', 'city']) and 'driv' not in scene_lower:
                setting = "outdoor urban street, city environment"
                lighting = "natural daylight"
                mood_desc = "city street atmosphere"
            
            # === INDOOR SCENES ===
            elif any(word in scene_lower for word in ['breakfast', 'kitchen', 'cooking']):
                setting = "cozy kitchen interior, home environment"
                lighting = "warm morning light"
                time_of_day = "morning"
                mood_desc = "cozy home atmosphere"
            elif any(word in scene_lower for word in ['cafe', 'coffee', 'restaurant']):
                setting = "cozy cafe interior, wooden tables, warm ambiance"
                lighting = "warm ambient lighting"
                mood_desc = "relaxed cafe atmosphere"
            elif any(word in scene_lower for word in ['exhibition', 'gallery', 'museum', 'art']):
                setting = "art gallery interior, paintings on walls"
                lighting = "spotlight with ambient"
                mood_desc = "cultural gallery atmosphere"
            elif any(word in scene_lower for word in ['office', 'work', 'meeting']):
                setting = "modern office interior"
                lighting = "fluorescent office lighting"
                mood_desc = "professional work environment"
            elif any(word in scene_lower for word in ['bedroom', 'sleep', 'night', 'dream']):
                setting = "bedroom interior, soft lighting"
                lighting = "dim ambient"
                time_of_day = "nighttime"
                mood_desc = "peaceful bedroom atmosphere"
            elif any(word in scene_lower for word in ['window', 'looking out', 'look out']) and not any(word in scene_lower for word in ['outdoor', 'bridge', 'rain']):
                setting = "indoor scene with window, view visible"
                lighting = "natural window light"
                mood_desc = "contemplative mood"
        
        # === ACTION DESCRIPTION ===
        # Extract the main action from the dialogue/action line
        action_words = []
        if any(word in scene_lower for word in ['walk', 'go', 'move', 'step', 'run']):
            action_words.append("walking")
        if any(word in scene_lower for word in ['sit', 'seat']):
            action_words.append("sitting")
        if any(word in scene_lower for word in ['look', 'watch', 'gaze', 'stare']):
            action_words.append("observing")
        if any(word in scene_lower for word in ['talk', 'speak', 'say', 'tell']):
            action_words.append("communicating")
        if any(word in scene_lower for word in ['eat', 'drink', 'consume']):
            action_words.append("enjoying meal")
        if any(word in scene_lower for word in ['read', 'book', 'newspaper']):
            action_words.append("reading")
        if any(word in scene_lower for word in ['sleep', 'rest', 'lie']):
            action_words.append("resting")
        
        if action_words:
            action_desc = "character " + ", ".join(action_words)
        else:
            action_desc = "character engaged in activity"
        
        return setting, lighting, time_of_day, mood_desc, action_desc
    
    def _determine_shot_type(self, scene_lower: str, is_first_scene: bool, is_last_scene: bool) -> str:
        """Determine shot type based on narrative context"""
        # Opening scene - wide establishing shot
        if is_first_scene:
            if any(word in scene_lower for word in ['home', 'kitchen', 'bedroom', 'cafe', 'station', 'street']):
                return "wide establishing"
            return "medium establishing"
        
        # Closing scene - medium shot
        if is_last_scene:
            return "medium close-up"
        
        # Action scenes - medium shots
        if any(word in scene_lower for word in ['walk', 'run', 'move', 'enter', 'exit', 'approach']):
            return "medium shot"
        
        # Intimate conversations
        if any(word in scene_lower for word in ['talk', 'speak', 'say', 'tell', 'ask']):
            return "medium close-up"
        
        # Default
        return "medium shot"
    
    def _extract_key_objects(self, scene_content: str, scene_lower: str) -> str:
        """Extract key objects mentioned in the scene that should be included in the prompt"""
        objects = []
        
        # Food and drink related
        if any(word in scene_lower for word in ['breakfast', 'lunch', 'dinner', 'food', 'meal', 'eat', 'eating']):
            if 'breakfast' in scene_lower:
                objects.append("breakfast on table")
            elif 'coffee' in scene_lower:
                objects.append("cup of coffee")
            else:
                objects.append("food on table")
        
        # Books and reading
        if any(word in scene_lower for word in ['book', 'read', 'reading', 'newspaper', 'magazine']):
            objects.append("holding a book")
        
        # Technology
        if any(word in scene_lower for word in ['phone', 'smartphone', 'laptop', 'computer', 'tablet']):
            objects.append("holding phone")
        
        # Musical instruments
        if any(word in scene_lower for word in ['guitar', 'piano', 'violin', 'music', 'instrument']):
            objects.append("musical instrument")
        
        # Toys (for children)
        if any(word in scene_lower for word in ['toy', 'toys', 'play', 'playing']):
            objects.append("toys on floor")
        
        # Art supplies
        if any(word in scene_lower for word in ['paint', 'painting', 'canvas', 'brush']):
            objects.append("art supplies")
        
        # Vegetables and cooking
        if any(word in scene_lower for word in ['vegetable', 'vegetables', 'cut', 'chop', 'cook', 'cooking', 'chef']):
            objects.append("cutting board with vegetables")
        
        # Food serving
        if 'serve' in scene_lower or 'dish' in scene_lower:
            objects.append("finished dish")
        
        return ", ".join(objects) if objects else ""
    
    def _get_characters_in_scene(self, scene_content: str, characters: List[str], scene_index: int = 0) -> List[str]:
        """
        Identify which characters appear in this scene.
        Handles multi-person scenes by checking for:
        - Explicit character name mentions
        - "and" conjunction (e.g., "Jack and Sara")
        - "with" construction (e.g., "Jack with Sara")
        - Pronouns after first scene (implying same characters)
        """
        scene_lower = scene_content.lower()
        present_chars = []
        
        # First pass: check for explicit character name mentions
        for char_name in characters:
            # Check if name appears in angle brackets or as standalone word
            if f'<{char_name.lower()}>' in scene_lower or f' {char_name.lower()} ' in scene_lower:
                present_chars.append(char_name)
            elif char_name.lower() in scene_lower and len(char_name) > 3:  # Avoid matching pronouns
                present_chars.append(char_name)
        
        # Second pass: check for "X and Y" or "X with Y" patterns
        if len(present_chars) == 0 and len(characters) > 1:
            # Check if script mentions multiple characters using "and" or "with"
            if ' and ' in scene_lower:
                # Try to match character pairs
                for i, char1 in enumerate(characters):
                    for char2 in characters[i+1:]:
                        if char1.lower() in scene_lower and char2.lower() in scene_lower:
                            if char1 not in present_chars:
                                present_chars.append(char1)
                            if char2 not in present_chars:
                                present_chars.append(char2)
            
            # Check for "with" pattern
            elif ' with ' in scene_lower:
                # First character in scene is the subject, "with" indicates companion
                if len(characters) >= 1:
                    present_chars.append(characters[0])
                    # The companion might be mentioned after "with"
                    for char in characters[1:]:
                        if char.lower() in scene_lower:
                            present_chars.append(char)
                            break
        
        # Third pass: for scenes after the first without explicit mentions,
        # assume all characters from the story are present
        if not present_chars and characters:
            # If it's a multi-character story and this isn't the first scene
            if scene_index > 0 and len(characters) > 1:
                present_chars = characters  # Assume all characters continue
            elif characters:
                present_chars = [characters[0]]
        
        return present_chars
    
    def _build_scene_char_description(self, char_names: List[str], char_appearances: Dict) -> str:
        """
        Build character description string for scene.
        CRITICAL FIX: Include character NAME first - this is crucial for SDXL to generate the correct person.
        """
        if not char_names:
            return "a person"
        
        descriptions = []
        for char_name in char_names:
            if char_name in char_appearances:
                app = char_appearances[char_name]
                gender = app.get('gender', 'male')
                age_cat = app.get('age_category', 'adult')
                
                # CRITICAL: Start with character NAME - this is essential for SDXL
                # Format: "Milo, a young boy with..."
                name_term = char_name.capitalize()  # Ensure proper capitalization
                
                # Build age term
                if age_cat == "child":
                    age_term = "young child" if gender == "female" else "young child"
                elif age_cat == "elderly":
                    age_term = "elderly person" if gender == "female" else "elderly person"
                else:
                    age_term = "young adult" if gender == "female" else "young adult"
                
                # Get distinctive features
                features = app.get('appearance_details', '')
                clothing = app.get('clothing', '')
                
                # Build description starting with name
                parts = [name_term]
                if features:
                    # Take first 2 features
                    feature_list = [f.strip() for f in features.split(',') if f.strip()]
                    if feature_list:
                        parts.append(feature_list[0])
                
                desc = "a " + ", ".join(parts)
                descriptions.append(desc)
        
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        elif len(descriptions) > 2:
            return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        else:
            return "people"
    
    def _rule_based_parse(self, parsed_script: Dict) -> str:
        raise NotImplementedError(
            "Rule-based parsing has been removed. LLM-only parsing enforced. "
            "Ensure Qwen model is cached and GPU is available."
        )

    def unload_model(self):
        """Explicitly unload model to free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[LocalQwen] Model unloaded, GPU memory freed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False


def create_qwen_parser(
    model_path: Optional[str] = None,
    use_quantization: bool = True
) -> LocalQwenParser:
    """Factory function to create optimized Qwen parser"""
    if model_path is None:
        from src.utils.mirror_config import get_models_cache_dir
        cache_base = get_models_cache_dir()
        qwen3_cached = cache_base / "models--Qwen--Qwen3-4B-Instruct-2507"

        if qwen3_cached.exists():
            model_path = "Qwen/Qwen3-4B-Instruct-2507"
            print(f"[Factory] Using cached Qwen3-4B-Instruct-2507 ({qwen3_cached})")
        else:
            raise RuntimeError(
                "[Factory] Qwen3-4B-Instruct-2507 not found in cache. "
                "LLM-only parsing enforced. Download the model first."
            )

    gpu_available = torch.cuda.is_available()
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_available else 0

    if gpu_vram_gb >= 16:
        config = {
            "model_name_or_path": model_path,
            "torch_dtype": torch.float16,
            "max_new_tokens": 2048
        }
        print(f"[Factory] Detected {gpu_vram_gb:.0f}GB VRAM -> FP16 mode")
    elif gpu_available:
        config = {
            "model_name_or_path": model_path,
            "torch_dtype": torch.float32,
            "max_new_tokens": 1536
        }
        print(f"[Factory] Detected {gpu_vram_gb:.0f}GB VRAM -> Quantization mode")
    else:
        config = {
            "model_name_or_path": model_path,
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "max_new_tokens": 1024
        }
        print("[Factory] No GPU detected -> CPU mode (slower)")

    return LocalQwenParser(**config)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Local Qwen2.5-7B Parser")
    print("=" * 70)

    test_file = str(Path(__file__).parent.parent.parent / "data" / "TaskA" / "06.txt")

    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        exit(1)

    from src.utils.mirror_config import setup_china_mirrors
    setup_china_mirrors()

    parser = create_qwen_parser(use_quantization=True)

    try:
        with parser:
            board = parser.process_script_file(test_file)

        print(f"\nParsing successful!")
        print(f"   Story ID: {board.story_id}")
        print(f"   Characters: {list(board.characters.keys())}")
        print(f"   Panels: {len(board.panels)}")
        print(f"   Style: {board.global_style}")

        output_path = "outputs/test_qwen_board.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        parser.save_production_board(board, output_path)
        print(f"   Saved to: {output_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
