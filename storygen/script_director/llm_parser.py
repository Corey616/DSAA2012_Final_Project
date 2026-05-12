"""
LLM Script Parser - Narrative director using large language models
Parses story scripts into structured production boards for visual generation
"""

import json
import re
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class Character:
    """Character information data class"""
    name: str
    visual_description: str
    token: str
    key_attributes: List[str] = field(default_factory=list)
    clothing: str = ""
    appearance_details: str = ""
    entity_type: str = "human"
    gender: str = ""
    age_bucket: str = ""
    wardrobe_palette: List[str] = field(default_factory=list)
    clothing_palette: List[str] = field(default_factory=list)
    face_palette: List[str] = field(default_factory=list)
    face_traits: List[str] = field(default_factory=list)


@dataclass
class Panel:
    """Story panel/scene data class"""
    panel_id: int
    raw_prompt: str
    enhanced_prompt: str = ""
    shot_type: str = "medium"
    camera_movement: str = "static"
    lighting_mood: str = "natural"
    composition: str = ""
    key_actions: List[str] = field(default_factory=list)
    interactions: List[Dict[str, str]] = field(default_factory=list)
    setting: str = ""
    time_of_day: str = "day"
    weather: str = ""
    key_objects: str = ""  # Key objects like breakfast, book, etc.

    def __post_init__(self):
        if self.key_actions is None:
            self.key_actions = []
        if self.interactions is None:
            self.interactions = []


@dataclass
class CharacterState:
    """Persistent state for a character across the whole story."""
    name: str
    identity_core: List[str] = field(default_factory=list)
    wardrobe_state: List[str] = field(default_factory=list)
    entity_type: str = "human"
    face_traits: List[str] = field(default_factory=list)
    wardrobe_palette: List[str] = field(default_factory=list)
    clothing_palette: List[str] = field(default_factory=list)
    face_palette: List[str] = field(default_factory=list)
    base_outfit: str = ""
    gender: str = ""
    age_bucket: str = ""
    emotion_state: str = ""
    pose_bias: str = ""
    visibility_rules: List[str] = field(default_factory=list)
    relations: Dict[str, str] = field(default_factory=dict)


@dataclass
class PanelEntityState:
    """Structured per-panel contract for one visible entity."""
    name: str
    present: bool = True
    entity_type: str = "human"
    role: str = ""
    action: str = ""
    expression: str = ""
    position: str = ""
    face_terms: List[str] = field(default_factory=list)
    wardrobe_terms: List[str] = field(default_factory=list)
    wardrobe_palette: List[str] = field(default_factory=list)
    clothing_palette: List[str] = field(default_factory=list)
    face_palette: List[str] = field(default_factory=list)
    gender: str = ""
    age_bucket: str = ""


@dataclass
class PanelState:
    """Structured rendering contract for a single panel."""
    panel_id: int
    characters_present: List[str] = field(default_factory=list)
    expected_count: int = 0
    count_confidence: str = "medium"
    action_beats: List[str] = field(default_factory=list)
    action_beat_sources: Dict[str, str] = field(default_factory=dict)
    spatial_layout: str = ""
    spatial_layout_source: str = ""
    scene_segment: str = ""
    camera_plan: Dict[str, str] = field(default_factory=dict)
    continuity_from_prev: List[str] = field(default_factory=list)
    must_show: List[str] = field(default_factory=list)
    must_show_sources: Dict[str, str] = field(default_factory=dict)
    must_not_show: List[str] = field(default_factory=list)
    local_constraints: List[str] = field(default_factory=list)
    emotion_cues: Dict[str, str] = field(default_factory=dict)
    panel_entities: List[PanelEntityState] = field(default_factory=list)
    reference_bindings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryState:
    """First-class structured story state used by render/eval stages."""
    story_id: str
    theme_style: str
    time_anchor: str = ""
    global_location_graph: List[str] = field(default_factory=list)
    global_constraints: List[str] = field(default_factory=list)
    narrative_goals: List[str] = field(default_factory=list)
    character_states: Dict[str, CharacterState] = field(default_factory=dict)
    persistent_props: List[str] = field(default_factory=list)
    scene_slots: List[str] = field(default_factory=list)
    stateful_objects: List[str] = field(default_factory=list)
    panel_states: List[PanelState] = field(default_factory=list)
    cross_panel_links: List[Dict[str, Any]] = field(default_factory=list)
    story_questions_for_eval: List[Dict[str, Any]] = field(default_factory=list)
    refinement_issues: List[Dict[str, Any]] = field(default_factory=list)
    refinement_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProductionBoard:
    """Complete story production blueprint"""
    story_id: str
    characters: Dict[str, Character]
    panels: List[Panel]
    global_style: str
    consistency_constraints: List[str] = field(default_factory=list)
    narrative_arc: str = "linear"
    story_state: Dict[str, Any] = field(default_factory=dict)
    render_plan: List[Dict[str, Any]] = field(default_factory=list)


class LLMScriptParser:
    """
    LLM-based script parser for narrative planning

    This module uses large language models to analyze story scripts and generate
    structured production boards containing character descriptions, scene plans,
    and visual instructions for downstream image generation.
    """

    SYSTEM_PROMPT = """You are a professional film director and storyboard artist.
Your task is to parse story scripts into detailed 'production blueprints' for multi-image generation systems.

## CRITICAL RULES - MUST FOLLOW:

### 1. STORY CONTEXT UNDERSTANDING (MOST IMPORTANT)
- pronouns like "He", "She", "They", "It" ALWAYS refer to characters/objects from PREVIOUS panels
- "He pauses at the door" → Understand WHICH door from context (door of current location)
- "She looks around" → She is in the SAME location as previous panel unless stated otherwise
- "It chases a ball" → "It" is the SAME animal/object from previous panels
- enhanced_prompt MUST include the setting/objects from previous panels when using pronouns
- DO NOT assume stories are about buses or trains unless explicitly stated

### 2. TIMELINE CONSISTENCY
- If the story takes place in the MORNING (breakfast, morning routine), ALL panels MUST have `time_of_day: "morning"`
- If the story takes place in the EVENING (dinner, evening routine), ALL panels MUST have `time_of_day: "evening"` or `"night"`
- NEVER change time_of_day between panels unless explicitly stated

### 3. SETTING CONSISTENCY
- If the story starts in a KITCHEN, subsequent panels should be KITCHEN or dining area
- If the story starts in a PARK, subsequent panels should remain in outdoor park settings
- Setting should flow naturally from the previous panel unless explicitly changed

### 4. VISUAL DESCRIPTION RULES
- visual_description MUST include SPECIFIC clothing (e.g., "blue button-up shirt", NOT "casual outfit")
- Include: hairstyle, hair color, eye color, build, skin tone, EXACT clothing description
- clothing field should match what's in visual_description

### 5. KEY OBJECTS CONSISTENCY
- Track key objects (book, ball, food, toys) across ALL panels
- If an object appears in Panel 1, it should appear/remain relevant in subsequent panels

### 6. OBJECT PROPAGATION (CRITICAL for continuity)
When a scene introduces a specific object (like "bus", "train", "car"), ALL subsequent
references to parts of that object MUST maintain the full context:
- Panel 1: "walks toward a bus" → Panel 2's "door" becomes "bus door"
- Panel 1: "enters a bus" → Panel 3's "window" becomes "bus window"
- Panel 1: "sits on a train" → Panel 2's "by the window" becomes "by the train window"
- NEVER drop the object context: "door" without context → "bus door", "train door", etc.

EXAMPLES of correct object propagation:
```
Input: [SCENE-1] <Ryan> walks quickly toward a bus.
       [SCENE-2] He pauses at the door and looks ahead.
       [SCENE-3] He gets inside and sits by the window.
Correct:
  Panel 1: "Ryan walks quickly toward a bus. ... near a bus."
  Panel 2: "Ryan pauses at the bus door and looks ahead. ... at the bus entrance."
  Panel 3: "Ryan gets inside the bus and sits by the window. ... inside the bus interior."

### 7. MULTI-CHARACTER SPATIAL LAYOUT (CRITICAL)
- For stories with 2+ characters, specify spatial positions:
  "Jack on the LEFT, Sara on the RIGHT"
- enhanced_prompt MUST mention BOTH characters' relative positions
- NEVER use generic phrases like "two individuals with distinct appearances"

### 8. CHARACTER DESCRIPTION UNIQUENESS
- Panel 1 enhanced_prompt: FULL character description
- Panels 2-3 enhanced_prompt: character NAME only (visual details are in the separate field)
- DO NOT repeat the full character description in EVERY panel's enhanced_prompt

### 9. CLOTHING CONSISTENCY
- A character's clothing MUST be identical across ALL panels
- clothing field must exactly match what's in visual_description
- If clothing changes, explicitly state the reason in the narrative

### 10. VALID time_of_day VALUES
- ONLY use these exact values: "morning", "afternoon", "evening", "night", "dawn", "dusk"
- NEVER use lighting moods or descriptions (e.g., "industrial lighting") for time_of_day
- lighting_mood field is SEPARATE from time_of_day

### 11. NON-HUMAN CHARACTER HANDLING
- For robots/machines: describe only mechanical features (metal, circuits, joints, LEDs)
- Do NOT add human attributes (hair, eye color, clothing, skin tone) to non-human characters
- For animals: use species-appropriate features (fur for mammals, feathers for birds, scales for reptiles)
- clothing field must be empty string "" for non-human characters

### 12. ENRICHED SCENE ANCHORING (CRITICAL)
- The enhanced_prompt MUST be at most 350 characters total — prefer concise, object-rich descriptions over long sentences
- The `enhanced_prompt` for EVERY panel MUST include at least 3 concrete visual details about the setting (objects, lighting details, background elements) embedded naturally in the description
- Generic scene descriptors like "in an airport" or "in a park" are INSUFFICIENT — always specify visible objects (e.g., "luggage carousels and departure screens" not just "airport terminal")
- Use the `must_show` field to list 2-4 specific objects or props that the image MUST contain
- The `setting` field MUST be specific enough that a reader can visualize the location without seeing the image

### 13. CHARACTER IDENTITY CONSISTENCY (CRITICAL)
- Every character's identity_core MUST include 4-5 DISTINCTIVE, visually-identifiable features that remain absolutely identical across ALL panels
- For humans: include specific hair color AND texture, eye color, skin tone, face shape or distinguishing features (freckles, glasses, specific nose/chin shape), approximate age
- For non-humans (animals, robots): include species-specific coloring, body type, texture/materials, distinctive markings
- The SAME identity terms MUST appear in EVERY panel's character description — do NOT add, remove, or modify descriptors between panels
- face_terms should be a strict subset of identity_core terms that are visible in close-up shots

Output Format:
Strict JSON format only, with the following structure. Do not include any explanations or markdown markers."""

    USER_PROMPT_TEMPLATE = """
Please deeply analyze the following story script and output a complete production blueprint JSON.

## Story Script:
```
{script_text}
```

## CRITICAL ANALYSIS:

### STEP 1: Identify Story Context
- What is the TIME OF DAY? (morning, afternoon, evening, night)
- Where does the story TAKE PLACE? (kitchen, office, street, etc.)
- What KEY OBJECTS appear? (breakfast food, book, coffee, toys, etc.)
- IMPORTANT: Track WHERE the story starts - this is the base setting for ALL panels!

### STEP 2: Pronoun Resolution (CRITICAL!)
When parsing scenes with pronouns (He/She/They/It):
- "He pauses at the door" → Understand context - what door? (current location's door)
- "She sits by the window" → Same room/location from previous panel
- "It chases a ball" → Same animal/object from previous panels
- enhanced_prompt MUST include the resolved context!

Example:
```
Script: [SCENE-1] <Ryan> walks in the park.
Script: [SCENE-2] He pauses at the door.
WRONG: "Ryan pauses at the door"
RIGHT: "Ryan pauses at the park entrance door"
```

### STEP 3: Character Analysis (characters)
Create detailed profiles for each character marked with <name>:
- `visual_description`: Detailed appearance (100+ chars), SPECIFIC clothing (e.g., "blue button-up shirt", NOT "casual outfit")
- `token`: Format "sks {{name}}" as unique identifier
- `key_attributes`: 3-5 most distinctive features (e.g., "short brown hair", "round glasses", "blue eyes")
- `clothing`: SPECIFIC clothing matching visual_description (e.g., "blue shirt and jeans")
- `appearance_details`: Specific details (hair color, eye color, skin tone)

### STEP 4: Panel/Scene Planning (panels)
CRITICAL: Setting and time MUST be consistent!

For each [SCENE]:
- `enhanced_prompt`: 150-200 char prompt including:
  * Character description (START with character name)
  * Main action (with resolved context for pronouns)
  * Setting (SAME as previous panel unless stated otherwise)
  * Key objects (SAME as previous panel)
  * Lighting matching time_of_day
  * "photorealistic, realistic photography, sharp focus, 8k detailed"
  
  **Example for morning routine story:**
  "Lily, a young woman with auburn hair and glasses, sits at kitchen table eating breakfast. Modern kitchen interior, morning sunlight through window. photorealistic..."

- `shot_type`: "extreme_closeup" / "closeup" / "medium" / "wide" / "over_shoulder" / "establishing"
- `camera_movement`: "static" / "slow_push_in" / "pull_back" / "pan_left_right" / "tracking"
- `lighting_mood`: MUST match time_of_day (morning = warm sunlight, evening = soft lamp light)
- `key_actions`: 2-4 specific visualizable actions
- `setting`: Detailed scene description (SAME as first panel unless changed!)
- `time_of_day`: SAME for ALL panels in same-time stories!
- `weather`: clear/rainy/snowy/cloudy
- `key_objects`: Track these across ALL panels (bus, food, book, etc.)

### STEP 5: Global Style (global_style)
Choose ONE style, MUST produce photorealistic images:
- "warm_cinematic_lifestyle" - Warm cinematic drama
- "urban_drama" - Urban drama
- "photorealistic_documentary" - Photorealistic documentary
- "cinematic_realistic" - Cinematic realistic

### STEP 6: Consistency Constraints (consistency_constraints)
List elements that must REMAIN CONSISTENT across ALL frames:
- Character appearance (hair color, clothing, features)
- Setting location (bus stop → bus interior, etc.)
- Time of day (ALL panels same time)
- Lighting style
- Key objects (food, book, toys, etc.)

### STEP 7: Object Propagation (CRITICAL!)
For each panel, check if there are objects from PREVIOUS panels that must be carried forward:
- If Panel 1 introduces "bus", Panel 2's "door" → "bus door"
- If Panel 1 introduces "train", Panel 3's "window" → "train window"
- enhanced_prompt MUST include the full context, not just "door" or "window"
- The `key_objects` field in each panel should INHERIT + ADD to the previous panel's objects

## OUTPUT EXAMPLE:
```json
{{
  "characters": {{...}},
  "panels": [
    {{
      "enhanced_prompt": "Lily, young woman with auburn hair, sitting at kitchen table eating breakfast. Modern kitchen interior, morning sunlight through window. photorealistic...",
      "time_of_day": "morning",
      "setting": "Modern kitchen with breakfast table",
      "key_objects": "breakfast food, coffee cup"
    }}
  ],
  "consistency_constraints": [
    "All panels must be morning time with warm sunlight",
    "All panels must be in or near kitchen setting"
  ]
}}
```
"""

    def __init__(self, llm_backend: str = "local", model_name: str = None):
        """
        Initialize the parser

        Args:
            llm_backend: LLM backend type ("local" | "api_openai" | "api_claude" | "api_deepseek")
            model_name: Model name (auto-selected if None)
        """
        self.llm_backend = llm_backend
        self.model_name = model_name or self._get_default_model()
        self.client = None
        self._initialize_client()

    def _get_default_model(self) -> str:
        """Get default model based on backend"""
        defaults = {
            "local": "llama3:70b",
            "api_openai": "gpt-4o",
            "api_claude": "claude-3-5-sonnet-20241022",
            "api_deepseek": "deepseek-v4-flash"  # Use via OpenCode Zen or direct API
        }
        return defaults.get(self.llm_backend, "gpt-4o")

    def _infer_gender_fallback(self, name: str) -> str:
        """Simple gender inference for fallback cases"""
        name_lower = name.lower()
        female_markers = {'girl', 'woman', 'female', 'lady', 'she', 'her', 'mom', 'nina', 'emma', 'sara', 'lily', 'olivia', 'rose'}
        male_markers = {'boy', 'man', 'male', 'he', 'his', 'dad', 'tom', 'jack', 'ben', 'leo', 'john', 'mike'}
        
        for marker in female_markers:
            if marker in name_lower:
                return "female"
        for marker in male_markers:
            if marker in name_lower:
                return "male"
        
        # Check name endings
        if name_lower.endswith(('a', 'e', 'i', 'y')):
            return "female"
        return "male"

    def _extract_char_from_raw_prompt(self, raw_prompt: str, char_names: List[str]) -> Optional[str]:
        """Extract character name from raw prompt like '<Lily> makes breakfast'"""
        # Match <Name> pattern
        match = re.search(r'<([A-Za-z]+)>', raw_prompt)
        if match:
            found_name = match.group(1)
            # Check if it's a known character
            for char_name in char_names:
                if char_name.lower() == found_name.lower():
                    return char_name
            return found_name
        return None

    def _initialize_client(self):
        """Initialize LLM client based on backend type"""
        if self.llm_backend == "local":
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
            except ImportError:
                print("[Parser] Warning: openai package not found, LLM calls will fail")
                self.client = None
        elif self.llm_backend == "api_openai":
            import openai
            self.client = openai.OpenAI()
        elif self.llm_backend == "api_deepseek":
            import openai
            # DeepSeek is OpenAI-compatible - uses the same API format
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            self.client = openai.OpenAI(
                base_url="https://api.deepseek.com",
                api_key=api_key,
            )
        elif self.llm_backend == "api_claude":
            import anthropic
            self.client = anthropic.Anthropic()

    def parse_raw_script(self, script_text: str) -> Dict[str, List[str]]:
        """
        Parse raw script text to extract scenes and characters

        Args:
            script_text: Raw script content

        Returns:
            dict: {"scenes": [...], "characters": [...], "raw_text": ...}
        """
        # Extract scenes using regex
        scene_pattern = r'\[SCENE-(\d+)\]\s*(.*?)(?=\[SEP\]|\Z)'
        scenes_raw = re.findall(scene_pattern, script_text, re.DOTALL | re.IGNORECASE)

        scenes = []
        characters_found = set()

        for scene_id, content in scenes_raw:
            clean_content = content.strip()
            scenes.append({
                "id": int(scene_id),
                "content": clean_content
            })

            # Extract character names marked with <name>
            char_pattern = r'<([^>]+)>'
            chars_in_scene = re.findall(char_pattern, clean_content)
            characters_found.update(chars_in_scene)

        return {
            "scenes": scenes,
            "characters": list(characters_found),
            "raw_text": script_text
        }

    def call_llm_for_analysis(self, parsed_script: Dict) -> str:
        """
        Call LLM for deep script analysis

        Returns:
            JSON string with analysis results
        """
        if self.client is None:
            # Fallback to rule-based parsing if no LLM available
            print("[Parser] Warning: No LLM client available, using rule-based fallback")
            return self._rule_based_parse(parsed_script)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            script_text=parsed_script["raw_text"]
        )

        if self.llm_backend in ["local", "api_openai", "api_deepseek"]:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Parser] LLM call failed: {e}, falling back to rule-based parsing")
                return self._rule_based_parse(parsed_script)

        elif self.llm_backend == "api_claude":
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return message.content[0].text
            except Exception as e:
                print(f"[Parser] LLM call failed: {e}, falling back to rule-based parsing")
                return self._rule_based_parse(parsed_script)

    def _rule_based_parse(self, parsed_script: Dict) -> str:
        """
        Fallback rule-based parsing when LLM is unavailable

        This creates a basic production board using pattern matching
        """
        characters = {}
        for char_name in parsed_script["characters"]:
            characters[char_name] = {
                "visual_description": f"A person named {char_name}",
                "token": f"sks {char_name}",
                "key_attributes": [],
                "clothing": "casual clothing",
                "appearance_details": "generic appearance"
            }

        panels = []
        for i, scene in enumerate(parsed_script["scenes"], 1):
            panels.append({
                "panel_id": i,
                "raw_prompt": scene['content'],
                "enhanced_prompt": f"{scene['content']}, high quality, detailed",
                "shot_type": "medium",
                "camera_movement": "static",
                "lighting_mood": "natural",
                "key_actions": [],
                "interactions": [],
                "setting": scene['content'][:100],
                "time_of_day": "day"
            })

        result = {
            "characters": characters,
            "panels": panels,
            "global_style": "cinematic_realistic",
            "consistency_constraints": list(parsed_script["characters"]),
            "narrative_arc": "linear"
        }

        return json.dumps(result)

    def _split_state_terms(self, text: str) -> List[str]:
        """Split a free-form string into de-duplicated state terms."""
        if not text:
            return []
        parts = re.split(r",|;|/|\band\b|\bwith\b", text)
        items = []
        seen = set()
        for part in parts:
            cleaned = re.sub(r"\s+", " ", part).strip(" .")
            if len(cleaned) < 3:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            items.append(cleaned)
        return items

    def _normalize_expression_label(self, text: str) -> str:
        """Map free-form emotional language into a compact expression vocabulary."""
        lowered = (text or "").lower()
        if any(token in lowered for token in ("smile", "grin", "laugh", "cheerful", "happy")):
            return "smiling"
        if any(token in lowered for token in ("surpris", "shock", "amazed", "astonish")):
            return "surprised"
        if any(token in lowered for token in ("worr", "concern", "anxious", "nervous")):
            return "worried"
        if any(token in lowered for token in ("sad", "cry", "tear", "upset", "downcast")):
            return "sad"
        if any(token in lowered for token in ("angry", "mad", "furious", "annoyed", "frustrat")):
            return "angry"
        if any(token in lowered for token in ("focus", "concentrat", "study", "read quietly", "write")):
            return "focused"
        if any(token in lowered for token in ("think", "wonder", "gaze", "looks out", "stares out", "reflect")):
            return "thoughtful"
        return ""

    def _extract_palette_terms(self, text: str) -> List[str]:
        """Extract stable clothing/identity color anchors."""
        color_terms = [
            "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
            "brown", "pink", "purple", "orange", "gold", "silver", "beige",
            "tan", "navy", "cream", "auburn", "blonde", "brunette", "ginger",
            "golden brown", "black and white", "white and brown",
        ]
        lowered = (text or "").lower()
        matches = []
        for term in color_terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
                matches.append(term)
        return list(dict.fromkeys(matches))

    # ── Specialized palette extraction ──────────────────────────────────

    _CLOTHING_COLORS = {
        "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
        "brown", "pink", "purple", "orange", "gold", "silver", "beige",
        "tan", "navy", "cream", "burgundy", "forest", "olive", "maroon",
        "teal", "coral", "khaki", "scarlet", "violet", "indigo", "ivory",
    }

    _FACE_COLORS = {
        "auburn", "blonde", "brunette", "ginger",
        "golden brown", "black and white", "white and brown",
        "strawberry", "chestnut", "raven",
    }

    def _extract_clothing_palette(self, text: str) -> List[str]:
        """Extract clothing-relevant colors from text."""
        if not text:
            return []
        found = []
        text_lower = text.lower()
        for color in self._CLOTHING_COLORS:
            if re.search(rf"(?<!\w){re.escape(color)}(?!\w)", text_lower):
                if color not in found:
                    found.append(color)
        return found

    def _extract_face_palette(self, text: str) -> List[str]:
        """Extract face/hair/eye colors from text."""
        if not text:
            return []
        found = []
        text_lower = text.lower()
        for color in self._CLOTHING_COLORS | self._FACE_COLORS:
            if re.search(rf"(?<!\w){re.escape(color)}(?!\w)", text_lower):
                if color not in found:
                    found.append(color)
        return found

    def _infer_gender_from_text_fields(self, *texts: str) -> str:
        """Infer gender only from character-local descriptive text, not story-global pronouns."""
        text = " ".join(str(text or "") for text in texts).lower()
        female_tokens = ("woman", "girl", "female", "lady", "she", "her")
        male_tokens = ("man", "boy", "male", "gentleman", "he", "his")
        has_female = any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in female_tokens)
        has_male = any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in male_tokens)
        if has_female and not has_male:
            return "female"
        if has_male and not has_female:
            return "male"
        return ""

    def _infer_gender_bucket(self, char_info: Character) -> str:
        """Infer a stable gender bucket from the preserved character fields."""
        explicit = getattr(char_info, "gender", "")
        if explicit:
            return explicit
        text = " ".join(
            [
                char_info.name,
                char_info.visual_description,
                char_info.appearance_details,
            ]
        ).lower()
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("woman", "girl", "female", "lady", "she", "her")):
            return "female"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("man", "boy", "male", "gentleman", "he", "his")):
            return "male"
        return "neutral"

    def _infer_age_bucket(self, char_info: Character) -> str:
        """Infer a stable age bucket from the preserved character fields."""
        explicit = getattr(char_info, "age_bucket", "")
        if explicit:
            return self._normalize_age_bucket_label(explicit)
        if getattr(char_info, "entity_type", "human") != "human":
            return ""
        text = " ".join(
            [
                char_info.visual_description,
                char_info.appearance_details,
                " ".join(char_info.key_attributes or []),
            ]
        ).lower()
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("baby", "infant")):
            return "baby"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("toddler",)):
            return "toddler"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("child", "kid", "boy", "girl", "school-age")):
            return "child"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("teen", "teenage", "adolescent")):
            return "teen"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) for token in ("elderly", "senior", "old", "aged")):
            return "elderly"
        if re.search(r"(?<!\w)young(?!\w)", text):
            return "young_adult"
        return "adult"

    def _extract_face_traits(self, char_info: Character) -> List[str]:
        """Keep only face-distinctive anchors for cross-panel identity consistency."""
        if getattr(char_info, "entity_type", "human") != "human":
            nonhuman_traits = self._extract_nonhuman_identity_terms(char_info, getattr(char_info, "entity_type", "human"))
            if nonhuman_traits:
                return nonhuman_traits
        existing = getattr(char_info, "face_traits", None) or []

        sources = []
        if existing:
            sources.extend([str(item) for item in existing])
        else:
            sources.extend(self._split_state_terms(char_info.appearance_details))
            sources.extend(self._split_state_terms(char_info.visual_description))
            sources.extend([str(item) for item in (char_info.key_attributes or [])])
        markers = (
            "hair", "eyes", "glasses", "ponytail", "braid", "beard", "mustache",
            "freckles", "scar", "jaw", "face", "cheek", "nose", "eyebrow",
        )
        traits = []
        used_categories = set()
        for source in sources:
            lowered = source.lower()
            if any(marker in lowered for marker in markers):
                if "hair" in lowered or "ponytail" in lowered or "braid" in lowered:
                    category = "hair"
                elif "eyes" in lowered:
                    category = "eyes"
                elif "glasses" in lowered:
                    category = "glasses"
                elif "beard" in lowered or "mustache" in lowered:
                    category = "facial_hair"
                elif "freckles" in lowered or "scar" in lowered:
                    category = "distinctive_mark"
                else:
                    category = lowered
                if category in used_categories:
                    continue
                used_categories.add(category)
                traits.append(source)
        return list(dict.fromkeys(traits[:4]))

    def _extract_base_outfit(self, char_info: Character) -> str:
        """Keep one stable outfit phrase for compiler-level anchoring."""
        outfit_terms = self._split_state_terms(char_info.clothing)
        if outfit_terms:
            return ", ".join(outfit_terms[:2])
        return self._split_state_terms(char_info.visual_description)[:1][0] if self._split_state_terms(char_info.visual_description) else ""

    def _stable_seed(self, text: str) -> int:
        """Generate a deterministic seed so fallback character templates stay stable across runs."""
        digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _detect_entity_type(self, name: str, *texts: str) -> str:
        """Classify entities so non-human prompts avoid leaking human-only labels."""
        combined = " ".join([name, *[text for text in texts if text]]).lower()
        robot_keywords = {
            "robot", "android", "mecha", "drone", "automaton", "cyborg",
            "machine", "droid", "mechanoid", "gynoid",
        }
        bird_keywords = {"bird", "owl", "eagle", "duck", "chicken", "penguin", "parrot", "sparrow", "crow"}
        reptile_keywords = {"snake", "lizard", "turtle", "frog", "gecko", "chameleon", "reptile"}
        fish_keywords = {"fish", "shark", "goldfish", "salmon", "trout", "dolphin", "whale"}
        mammal_keywords = {
            "dog", "cat", "rabbit", "horse", "lion", "tiger", "bear", "wolf", "fox",
            "deer", "elephant", "monkey", "panda", "koala", "puppy", "kitten",
            "cow", "sheep", "goat", "pig",
        }

        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", combined) for token in robot_keywords):
            return "robot"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", combined) for token in bird_keywords):
            return "bird"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", combined) for token in reptile_keywords):
            return "reptile"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", combined) for token in fish_keywords):
            return "fish"
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", combined) for token in mammal_keywords):
            return "mammal"
        return "human"

    def _infer_story_pronoun_gender(self, raw_text: str) -> str:
        """Use global pronoun evidence only when it is one-sided and reliable."""
        lowered = (raw_text or "").lower()
        has_male = bool(re.search(r"(?<!\w)(he|his|him)(?!\w)", lowered))
        has_female = bool(re.search(r"(?<!\w)(she|her|hers)(?!\w)", lowered))
        if has_male and not has_female:
            return "male"
        if has_female and not has_male:
            return "female"
        return ""

    def _normalize_age_bucket_label(self, value: str) -> str:
        lowered = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "infant": "baby",
            "newborn": "baby",
            "baby": "baby",
            "toddler": "toddler",
            "kid": "child",
            "kids": "child",
            "child": "child",
            "children": "child",
            "school_age": "child",
            "school_aged": "child",
            "teen": "teen",
            "teenager": "teen",
            "teenage": "teen",
            "young": "young_adult",
            "young_adult": "young_adult",
            "adult": "adult",
            "elder": "elderly",
            "elderly": "elderly",
            "senior": "elderly",
        }
        return mapping.get(lowered, lowered)

    def _extract_nonhuman_identity_terms(self, char_info: Character, entity_type: str) -> List[str]:
        """Recover species/material anchors that are more useful than human face descriptors."""
        import re

        marker_map = {
            "robot": ("robot", "metal", "mechanical", "steel", "chrome", "sensor", "led", "circuit", "joint", "chassis"),
            "bird": ("bird", "feather", "plumage", "wing", "beak", "talon"),
            "mammal": ("fur", "coat", "pelt", "whisker", "paw", "tail", "ear", "mane", "snout", "stripe", "spot"),
            "reptile": ("scale", "shell", "reptile", "crest", "hide"),
            "fish": ("scale", "fin", "gill", "streamlined", "aquatic"),
        }
        markers = marker_map.get(entity_type, ())
        if not markers:
            return []
        sources = []
        sources.extend(self._split_state_terms(char_info.appearance_details))
        sources.extend(self._split_state_terms(char_info.visual_description))
        sources.extend([str(item) for item in (char_info.key_attributes or [])])
        traits = []
        for source in sources:
            cleaned_source = re.sub(r"^(a|an|the)\s+", "", source.strip(), flags=re.IGNORECASE)
            lowered = cleaned_source.lower()
            preferred_phrases = [
                "articulated limbs", "visible mechanical joints", "optical sensor array", "glowing led indicators",
                "chrome finish", "brushed steel", "matte black metal", "smooth feathers", "glossy plumage",
                "sleek wings", "soft downy feathers", "fluffy fur", "shiny coat", "patterned scales",
                "shiny scales", "translucent fins",
            ]
            for phrase in preferred_phrases:
                if phrase in lowered:
                    cleaned_source = phrase
                    lowered = cleaned_source
                    break
            if any(marker in lowered for marker in markers):
                traits.append(cleaned_source)
        return list(dict.fromkeys(traits[:4]))

    def _normalize_human_age_bucket(self, name: str, raw_text: str, existing_value: str) -> str:
        """Keep child labels tight so toy/floor cues do not spill into adult stories."""
        normalized_existing = self._normalize_age_bucket_label(existing_value)
        name_lower = (name or "").lower()
        raw_lower = (raw_text or "").lower()

        explicit_markers = (
            ("baby", "baby"),
            ("infant", "baby"),
            ("toddler", "toddler"),
            ("child", "child"),
            ("kid", "child"),
            ("boy", "child"),
            ("girl", "child"),
            ("teen", "teen"),
            ("elder", "elderly"),
            ("senior", "elderly"),
        )
        for marker, bucket in explicit_markers:
            if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", name_lower):
                return bucket

        child_indicators = [
            "toy", "toys", "crib", "stroller", "car seat", "infant", "baby", "toddler",
            "nap time", "nursery", "play mat",
        ]
        adult_indicators = [
            "office", "meeting", "work", "job", "computer", "keyboard", "types quickly", "commute",
        ]
        child_score = sum(1 for indicator in child_indicators if indicator in raw_lower)
        adult_score = sum(1 for indicator in adult_indicators if indicator in raw_lower)

        if "baby" in raw_lower or "infant" in raw_lower:
            return "baby"
        if "toddler" in raw_lower:
            return "toddler"
        if normalized_existing in {"baby", "toddler", "child", "teen", "elderly"}:
            return normalized_existing
        if normalized_existing in {"adult", "young_adult"} and child_score >= 1 and adult_score == 0:
            return "child"
        if normalized_existing in {"adult", "young_adult"}:
            return normalized_existing
        if child_score >= 1 and adult_score == 0:
            return "child"
        return normalized_existing or "adult"

    def _normalize_character_entity(self, name: str, char_info: Character, raw_text: str) -> None:
        """Single normalization choke point after LLM merge, before story state is built."""
        original_age_bucket = self._normalize_age_bucket_label(char_info.age_bucket)
        entity_type = self._detect_entity_type(
            name,
            char_info.visual_description,
            char_info.appearance_details,
            char_info.clothing,
            " ".join(char_info.key_attributes or []),
        )
        char_info.entity_type = entity_type

        if entity_type != "human":
            char_info.gender = ""
            char_info.age_bucket = ""
            char_info.clothing = ""
            char_info.wardrobe_palette = []
            char_info.clothing_palette = []
            char_info.face_palette = []
            nonhuman_terms = self._extract_nonhuman_identity_terms(char_info, entity_type)
            if nonhuman_terms:
                char_info.face_traits = nonhuman_terms
            return

        char_info.age_bucket = self._normalize_human_age_bucket(name, raw_text, char_info.age_bucket)
        pronoun_gender = self._infer_story_pronoun_gender(raw_text)
        current_gender = (char_info.gender or "").strip().lower()
        description_gender = self._infer_gender_from_text_fields(
            char_info.visual_description,
            char_info.appearance_details,
            " ".join(char_info.key_attributes or []),
        )
        inferred_gender = ""
        infer_gender_helper = getattr(self, "_infer_gender", None)
        if callable(infer_gender_helper):
            try:
                inferred_gender = (infer_gender_helper(name) or "").strip().lower()
            except Exception:
                inferred_gender = ""

        if description_gender in {"male", "female"}:
            char_info.gender = description_gender
        elif pronoun_gender and current_gender not in {"male", "female"}:
            char_info.gender = pronoun_gender
        elif inferred_gender in {"male", "female"}:
            if current_gender not in {"male", "female"}:
                char_info.gender = inferred_gender
            elif not pronoun_gender and current_gender != inferred_gender:
                char_info.gender = inferred_gender

        if char_info.age_bucket in {"baby", "toddler", "child"}:
            lowered_clothing = (char_info.clothing or "").lower()
            adult_clothing_tokens = ("blouse", "skirt", "dress", "suit", "office", "leather jacket", "heels")
            if any(token in lowered_clothing for token in adult_clothing_tokens):
                if char_info.age_bucket == "baby":
                    char_info.clothing = "soft baby clothes"
                else:
                    char_info.clothing = "colorful children's clothes"
            elif char_info.age_bucket in {"toddler", "child"} and char_info.clothing and not any(
                token in lowered_clothing for token in ("kid", "child", "baby", "toddler")
            ):
                char_info.clothing = f"kid-sized {char_info.clothing}"

    def _infer_entity_positions(
        self,
        characters_present: List[str],
        spatial_layout: str,
    ) -> Dict[str, str]:
        """Assign simple per-entity positions from the panel layout text."""
        positions: Dict[str, str] = {}
        if not characters_present:
            return positions
        lowered = (spatial_layout or "").lower()
        if len(characters_present) >= 2 and ("left" in lowered and "right" in lowered):
            positions[characters_present[0]] = "left"
            positions[characters_present[1]] = "right"
        elif "center" in lowered or "middle" in lowered:
            positions[characters_present[0]] = "center"
        elif "behind" in lowered and len(characters_present) >= 2:
            positions[characters_present[0]] = "front"
            positions[characters_present[1]] = "behind"
        return positions

    def _infer_panel_entity_states(
        self,
        characters_present: List[str],
        characters: Dict[str, Character],
        action_beats: List[str],
        emotion_cues: Dict[str, str],
        spatial_layout: str,
    ) -> List[PanelEntityState]:
        """Build one explicit entity contract per visible character."""
        entity_states: List[PanelEntityState] = []
        positions = self._infer_entity_positions(characters_present, spatial_layout)
        shared_action = action_beats[0] if action_beats else ""
        for index, char_name in enumerate(characters_present):
            if char_name not in characters:
                continue
            char_info = characters[char_name]
            wardrobe_terms = self._split_state_terms(char_info.clothing)
            entity_action = shared_action if len(characters_present) > 1 else (
                action_beats[index] if index < len(action_beats) else shared_action
            )
            entity_states.append(
                PanelEntityState(
                    name=char_name,
                    present=True,
                    entity_type=getattr(char_info, "entity_type", "human"),
                    role="primary" if index == 0 else "supporting",
                    action=entity_action,
                    expression=emotion_cues.get(char_name, ""),
                    position=positions.get(char_name, ""),
                    face_terms=self._extract_face_traits(char_info),
                    wardrobe_terms=wardrobe_terms[:3],
                    wardrobe_palette=self._extract_palette_terms(" ".join(wardrobe_terms) or char_info.visual_description),
                    clothing_palette=self._extract_clothing_palette(" ".join(wardrobe_terms) if wardrobe_terms else ""),
                    face_palette=self._extract_face_palette(char_info.visual_description or ""),
                    gender=self._infer_gender_bucket(char_info),
                    age_bucket=self._infer_age_bucket(char_info),
                )
            )
        return entity_states

    def _infer_panel_emotion_cues(
        self,
        panel: Panel,
        characters_present: List[str],
    ) -> Dict[str, str]:
        """Infer compact, panel-local expression cues from the scene text."""
        text = " ".join(
            [
                panel.raw_prompt or "",
                panel.enhanced_prompt or "",
                ", ".join(panel.key_actions or []),
            ]
        ).strip()
        if not text:
            return {}

        label = self._normalize_expression_label(text)
        if not label or not characters_present:
            return {}

        return {
            char_name: label
            for char_name in characters_present
        }

    def _extract_panel_characters(
        self,
        panel: Panel,
        characters: Dict[str, Character],
        previous_chars: Optional[List[str]] = None,
    ) -> List[str]:
        """Infer which characters should be visible in a panel."""
        text = f"{panel.raw_prompt} {panel.enhanced_prompt}".lower()
        present = []
        for char_name in characters.keys():
            lower_name = char_name.lower()
            if f"<{lower_name}>" in text or lower_name in text:
                present.append(char_name)

        if present:
            return list(dict.fromkeys(present))

        all_names = list(characters.keys())
        if len(all_names) == 1:
            return all_names
        prior_chars = previous_chars[:] if previous_chars else []
        plural_markers = ("they", "them", "their", "both", "together", "each other")
        singular_markers = ("he ", "she ", "his ", "her ", "him ", "it ", "its ")

        if any(token in text for token in plural_markers):
            if prior_chars:
                return list(dict.fromkeys(prior_chars))
            return all_names
        if any(token in text for token in singular_markers):
            if len(prior_chars) == 1:
                return prior_chars
            return prior_chars[:1] or all_names[:1]
        if len(prior_chars) == 1:
            return prior_chars
        return all_names[:1]

    def _extract_spatial_layout_with_source(
        self,
        raw_text: str,
        enhanced_text: str = "",
    ) -> Tuple[str, str]:
        """Pull out a simple spatial layout phrase and remember where it came from."""
        for source_name, text in (
            ("raw_prompt", raw_text),
            ("enhanced_prompt", enhanced_text),
        ):
            if not text:
                continue
            lower_text = text.lower()
            layout_phrases = []
            for phrase in (
                "on the left", "on the right", "left side", "right side",
                "in the center", "in the middle", "in front of", "behind",
                "next to each other", "side by side", "facing each other",
            ):
                if phrase in lower_text:
                    layout_phrases.append(phrase)
            if layout_phrases:
                return ", ".join(dict.fromkeys(layout_phrases)), source_name
        return "", ""

    def _extract_spatial_layout(self, text: str) -> str:
        """Backward-compatible wrapper for older callers."""
        return self._extract_spatial_layout_with_source(text)[0]

    def _infer_panel_props(self, panel: Panel) -> List[str]:
        """Infer panel-local objects and relation targets beyond parser key_objects."""
        props = self._split_state_terms(panel.key_objects)
        text = f"{panel.raw_prompt} {panel.enhanced_prompt} {panel.setting}".lower()

        def indicator_matches(indicator: str, haystack: str) -> bool:
            normalized = re.escape(indicator.lower())
            return re.search(rf"(?<!\w){normalized}(?!\w)", haystack) is not None

        phrase_candidates = [
            ("bus door", ("bus door", "door of the bus", "at the door", "bus entrance")),
            ("train door", ("train door", "door of the train", "platform door")),
            ("window seat", ("window seat", "seat by the window", "sits by the window")),
            ("book", ("book",)),
            ("toys", ("toy", "toys")),
            ("gloves", ("glove", "gloves")),
            ("car seat", ("car seat", "baby seat", "child seat")),
            ("branch", ("branch", "perch")),
            ("spatula", ("spatula",)),
            ("stove", ("stove",)),
            ("breakfast food", ("breakfast", "food")),
            ("bench", ("bench",)),
            ("table", ("table",)),
            ("chair", ("chair",)),
            ("window", ("window",)),
            ("door", ("door",)),
            ("bus", ("bus",)),
            ("train", ("train",)),
            ("car", ("car",)),
            ("seat", ("seat",)),
            ("screen", ("screen",)),
            ("exhibit", ("exhibit", "display case", "labels")),
            ("painting", ("painting",)),
            ("canvas", ("canvas",)),
            ("soccer ball", ("soccer ball",)),
            ("ball", ("ball",)),
            ("suitcase", ("suitcase",)),
            ("map", ("map",)),
            ("platter", ("platter",)),
            ("prepared food", ("prepared food", "finished dish")),
            ("vegetables", ("vegetables",)),
            ("cutting board", ("cutting board",)),
            ("notebook", ("notebook",)),
            ("bag", ("bag", "backpack")),
            ("clock", ("clock",)),
        ]

        seen = {prop.lower() for prop in props}
        for label, indicators in phrase_candidates:
            if label in seen:
                continue
            if any(indicator_matches(indicator, text) for indicator in indicators):
                props.append(label)
                seen.add(label)

        return props[:6]

    def _extract_critical_panel_props(self, panel: Panel, characters_present: List[str], characters: Dict[str, Character]) -> List[str]:
        """Promote small story-critical props so they survive prompt dedupe and generic-scene suppression."""
        text = f"{panel.raw_prompt} {panel.enhanced_prompt} {panel.setting} {panel.key_objects}".lower()
        critical_props = []
        prop_checks = [
            ("window", ("window",)),
            ("chair", ("chair",)),
            ("book", ("book",)),
            ("gloves", ("glove", "gloves")),
            ("toys", ("toy", "toys")),
            ("car seat", ("car seat", "baby seat", "child seat")),
            ("branch", ("branch", "perch")),
        ]
        for label, indicators in prop_checks:
            if any(indicator in text for indicator in indicators):
                critical_props.append(label)

        panel_has_child = any(
            getattr(characters.get(name), "age_bucket", "") in {"baby", "toddler", "child"}
            for name in characters_present
        )
        if panel_has_child and "car" in text and "car seat" not in critical_props:
            critical_props.append("car seat")

        return list(dict.fromkeys(critical_props))

    def _infer_action_beat_items(self, panel: Panel) -> List[Dict[str, str]]:
        """Extract compact visible actions and keep their provenance."""
        def normalize_action_verb(word: str, action_roots: set) -> str:
            lowered = word.lower().strip(" ,.")
            candidates = [lowered]
            if lowered.endswith("ies") and len(lowered) > 3:
                candidates.append(lowered[:-3] + "y")
            if lowered.endswith("ing") and len(lowered) > 4:
                candidates.extend([lowered[:-3], lowered[:-3] + "e"])
            if lowered.endswith("ed") and len(lowered) > 3:
                candidates.extend([lowered[:-2], lowered[:-1]])
            if lowered.endswith("es") and len(lowered) > 3:
                candidates.extend([lowered[:-2], lowered[:-1]])
            if lowered.endswith("s") and len(lowered) > 2:
                candidates.append(lowered[:-1])
            for candidate in candidates:
                if candidate in action_roots:
                    return candidate
            return lowered

        action_roots = {
            "walk", "run", "sit", "stand", "look", "watch", "wait", "pause", "get",
            "enter", "exit", "arrive", "leave", "hold", "carry", "take", "place",
            "open", "close", "turn", "read", "write", "play", "eat", "drink",
            "talk", "smile", "laugh", "stop", "continue", "fall", "roll", "rest",
            "drive", "meet", "scan", "work", "study", "check", "pull", "push",
            "kick", "jump", "rest", "lie", "move", "dust", "paint", "cut",
            "taste", "serve", "pack", "think", "observe", "chat", "approach",
            "stretch", "cross", "type", "lean", "relax",
            "board", "cook", "make", "nod", "point", "gesture",
        }
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "of", "for", "with", "by",
            "through", "from", "into", "near", "inside", "outside",
        }
        generic_actions = {
            "look", "talk", "play", "walk", "move", "sit", "stand",
            "waiting", "walking", "standing", "looking",
        }
        relation_markers = {
            "at", "by", "with", "toward", "towards", "into", "inside", "outside",
            "beside", "near", "through", "from", "window", "door", "book", "bus",
            "table", "bench", "seat", "ball", "screen", "exhibit", "canvas",
        }
        appearance_markers = {
            "hair", "eyes", "skin", "coat", "jacket", "pants", "shirt", "scarf",
            "boots", "dress", "blouse", "sweater", "hoodie", "ponytail", "wearing",
            "blonde", "auburn", "dark hair", "light hair", "fair skin",
        }

        def extract_action_candidates(text: str) -> List[str]:
            text = text or ""
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            clauses = re.split(r"[.;]|,| and | then | while ", text, flags=re.IGNORECASE)
            phrases = []
            for clause in clauses:
                words = clause.strip().split()
                if not words:
                    continue
                for idx, word in enumerate(words):
                    root = normalize_action_verb(word, action_roots)
                    if root not in action_roots:
                        continue
                    phrase_words = []
                    for candidate in words[idx:idx + 7]:
                        lowered = candidate.lower()
                        if phrase_words and lowered in {"and", "then", "but"}:
                            break
                        phrase_words.append(candidate)
                    phrase = " ".join(phrase_words).strip(" ,.")
                    if phrase:
                        phrases.append(phrase)
                    break
            return phrases

        def canonical_action_key(phrase: str) -> str:
            for token in phrase.split():
                root = normalize_action_verb(token, action_roots)
                if root in action_roots:
                    return root
            return phrase.lower()

        def specificity_score(phrase: str) -> int:
            lowered = phrase.lower().strip(" ,.")
            words = lowered.split()
            score = len(words)
            if any(word in relation_markers for word in words):
                score += 2
            if words and words[-1] in stop_words:
                score -= 2
            if lowered in generic_actions:
                score -= 2
            return score

        candidate_pool: List[Dict[str, Any]] = []
        rank = 0
        for phrase in extract_action_candidates(panel.raw_prompt):
            candidate_pool.append({"text": phrase, "source": "raw_prompt", "rank": rank})
            rank += 1
        for action in (panel.key_actions or []):
            if str(action).strip():
                candidate_pool.append({"text": str(action).strip(), "source": "key_actions", "rank": rank})
                rank += 1
        if not candidate_pool:
            for phrase in extract_action_candidates(panel.enhanced_prompt):
                candidate_pool.append({"text": phrase, "source": "enhanced_prompt", "rank": rank})
                rank += 1

        source_priority = {
            "raw_prompt": 0,
            "key_actions": 1,
            "enhanced_prompt": 2,
        }
        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in sorted(
            candidate_pool,
            key=lambda candidate: (
                source_priority.get(candidate["source"], 9),
                candidate["rank"],
                -specificity_score(candidate["text"]),
            ),
        ):
            phrase = item["text"]
            lowered = phrase.lower()
            action_key = canonical_action_key(lowered)
            if action_key == lowered and any(marker in lowered for marker in appearance_markers):
                continue
            if action_key in seen:
                continue
            seen.add(action_key)
            cleaned = " ".join(
                token for token in lowered.split()
                if token not in {"he", "she", "they", "it", "his", "her", "their"}
            ).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
            if not cleaned:
                continue
            meaningful = [
                token for token in cleaned.split()
                if token not in stop_words
            ]
            if not meaningful:
                continue
            if cleaned.split()[-1] in stop_words and len(cleaned.split()) <= 1:
                continue
            deduped.append({
                "text": cleaned,
                "source": item["source"],
            })
        return deduped[:3]

    def _infer_action_beats(self, panel: Panel) -> List[str]:
        """Extract a compact list of visible actions for the panel."""
        return [item["text"] for item in self._infer_action_beat_items(panel)]

    def _build_story_state(
        self,
        story_id: str,
        raw_text: str,
        characters: Dict[str, Character],
        panels: List[Panel],
        global_style: str,
        consistency_constraints: List[str],
        narrative_arc: str,
    ) -> StoryState:
        """Derive a first-class story state from the parsed production board."""
        def contains_term(term: str, haystack: str) -> bool:
            return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", haystack.lower()) is not None

        time_counts = {}
        location_graph = []
        prop_counts = {}
        stateful_objects = set()

        for panel in panels:
            time_key = (panel.time_of_day or "").strip().lower()
            if time_key:
                time_counts[time_key] = time_counts.get(time_key, 0) + 1

            setting = (panel.setting or "").strip()
            if setting and (not location_graph or location_graph[-1] != setting):
                location_graph.append(setting)

            for prop in self._infer_panel_props(panel):
                lowered = prop.lower()
                prop_counts[lowered] = prop_counts.get(lowered, 0) + 1

            panel_text = f"{panel.raw_prompt} {panel.enhanced_prompt}".lower()
            for noun in (
                "door", "window", "table", "book", "ball", "cup",
                "bus", "train", "car", "bench", "plate", "phone",
            ):
                if contains_term(noun, panel_text):
                    stateful_objects.add(noun)

        time_anchor = max(time_counts, key=time_counts.get) if time_counts else ""
        sorted_props = sorted(prop_counts.items(), key=lambda item: (-item[1], item[0]))
        persistent_props = [prop for prop, count in sorted_props if count > 1]
        if not persistent_props:
            persistent_props = [prop for prop, _ in sorted_props[:4]]

        character_states = {}
        emotion_counts = {char_name: {} for char_name in characters.keys()}
        for char_name, char_info in characters.items():
            identity_core = self._split_state_terms(char_info.appearance_details)
            if not identity_core:
                identity_core = self._split_state_terms(char_info.visual_description)[:4]

            wardrobe_state = self._split_state_terms(char_info.clothing)
            face_traits = self._extract_face_traits(char_info)
            clothing_palette = self._extract_clothing_palette(char_info.clothing)
            face_palette = self._extract_face_palette(
                f"{char_info.visual_description} {char_info.appearance_details}"
            )
            # Keep wardrobe_palette as backward-compatible union
            wardrobe_palette = list(set(clothing_palette) | set(face_palette))
            visibility_rules = ["identity_core must remain stable across panels"]
            if wardrobe_state:
                visibility_rules.append("wardrobe_state must remain stable across panels")

            relations = {}
            for panel in panels:
                for interaction in panel.interactions:
                    target = interaction.get("target")
                    relation = interaction.get("type") or interaction.get("relation")
                    if target and relation:
                        relations[target] = relation

            character_states[char_name] = CharacterState(
                name=char_name,
                identity_core=identity_core,
                wardrobe_state=wardrobe_state,
                entity_type=getattr(char_info, "entity_type", "human"),
                face_traits=face_traits,
                wardrobe_palette=wardrobe_palette,
                clothing_palette=clothing_palette,
                face_palette=face_palette,
                base_outfit=self._extract_base_outfit(char_info),
                gender=self._infer_gender_bucket(char_info),
                age_bucket=self._infer_age_bucket(char_info),
                emotion_state="",
                pose_bias="",
                visibility_rules=visibility_rules,
                relations=relations,
            )

        panel_states = []
        cross_panel_links = []
        story_questions = []
        previous_panel = None
        previous_props = []
        previous_chars = []
        scene_segment_index = 0

        for index, panel in enumerate(panels):
            current_props = self._infer_panel_props(panel)
            characters_present = self._extract_panel_characters(
                panel,
                characters,
                previous_chars=previous_chars,
            )
            action_beat_items = self._infer_action_beat_items(panel)
            action_beats = [item["text"] for item in action_beat_items]
            action_beat_sources = {
                item["text"]: item["source"]
                for item in action_beat_items
            }
            spatial_layout, spatial_layout_source = self._extract_spatial_layout_with_source(
                panel.raw_prompt,
                panel.enhanced_prompt,
            )
            if spatial_layout_source != "raw_prompt":
                spatial_layout = ""
                spatial_layout_source = ""
            emotion_cues = self._infer_panel_emotion_cues(panel, characters_present)
            panel_entities = self._infer_panel_entity_states(
                characters_present=characters_present,
                characters=characters,
                action_beats=action_beats,
                emotion_cues=emotion_cues,
                spatial_layout=spatial_layout,
            )
            critical_props = self._extract_critical_panel_props(panel, characters_present, characters)
            current_props = list(dict.fromkeys(current_props + critical_props))
            for char_name, emotion in emotion_cues.items():
                counts = emotion_counts.setdefault(char_name, {})
                counts[emotion] = counts.get(emotion, 0) + 1

            if previous_panel is None or panel.setting != previous_panel.setting:
                scene_segment_index += 1
            scene_segment = f"scene_{scene_segment_index}"

            continuity_from_prev = []
            if previous_panel is not None:
                if panel.setting and previous_panel.setting:
                    if panel.setting == previous_panel.setting:
                        continuity_from_prev.append(f"remain in {panel.setting}")
                    else:
                        continuity_from_prev.append(
                            f"transition from {previous_panel.setting} to {panel.setting}"
                        )

                shared_props = [prop for prop in current_props if prop.lower() in {p.lower() for p in previous_props}]
                if shared_props:
                    continuity_from_prev.append(f"carry over {', '.join(shared_props[:3])}")
                elif previous_props:
                    carry = [prop for prop in previous_props if prop.lower() in persistent_props]
                    if carry:
                        continuity_from_prev.append(f"carry over {', '.join(carry[:3])}")

                if panel.time_of_day and previous_panel.time_of_day and panel.time_of_day == previous_panel.time_of_day:
                    continuity_from_prev.append(f"same {panel.time_of_day} time anchor")

            must_show: List[str] = []
            must_show_sources: Dict[str, str] = {}

            def add_must_show(item: str, source: str) -> None:
                cleaned = str(item or "").strip()
                if not cleaned or cleaned in must_show:
                    return
                must_show.append(cleaned)
                must_show_sources[cleaned] = source

            for char_name in characters_present:
                add_must_show(char_name, "character")
            if action_beats:
                add_must_show(action_beats[0], action_beat_sources.get(action_beats[0], "action"))
            if spatial_layout:
                add_must_show(spatial_layout, spatial_layout_source or "layout")
            if panel.setting:
                add_must_show(panel.setting, "setting")
            for item in critical_props:
                add_must_show(item, "critical_prop")
            for item in current_props[:3]:
                add_must_show(item, "panel_prop")

            local_constraints = []
            if panel.time_of_day:
                local_constraints.append(f"time_of_day={panel.time_of_day}")
            if panel.lighting_mood:
                local_constraints.append(f"lighting={panel.lighting_mood}")
            if panel.weather:
                local_constraints.append(f"weather={panel.weather}")

            panel_state = PanelState(
                panel_id=panel.panel_id,
                characters_present=characters_present,
                expected_count=len(characters_present),
                count_confidence="high" if characters_present and any(name.lower() in f"{panel.raw_prompt} {panel.enhanced_prompt}".lower() for name in characters_present) else ("medium" if characters_present else "low"),
                action_beats=action_beats,
                action_beat_sources=action_beat_sources,
                spatial_layout=spatial_layout,
                spatial_layout_source=spatial_layout_source,
                scene_segment=scene_segment,
                camera_plan={
                    "shot_type": panel.shot_type,
                    "camera_movement": panel.camera_movement,
                    "lighting_mood": panel.lighting_mood,
                },
                continuity_from_prev=continuity_from_prev,
                must_show=must_show,
                must_show_sources=must_show_sources,
                must_not_show=[],
                local_constraints=local_constraints,
                emotion_cues=emotion_cues,
                panel_entities=panel_entities,
                reference_bindings={
                    "character_tokens": [
                        characters[name].token for name in characters_present if name in characters
                    ],
                    "scene_segment": scene_segment,
                },
            )
            panel_states.append(panel_state)

            if previous_panel is not None:
                cross_panel_links.append({
                    "from_panel": previous_panel.panel_id,
                    "to_panel": panel.panel_id,
                    "scene_segment": scene_segment,
                    "same_scene_segment": previous_panel.setting == panel.setting,
                    "location_transition": (
                        f"{previous_panel.setting} -> {panel.setting}"
                        if previous_panel.setting or panel.setting else ""
                    ),
                    "carry_over_props": [
                        prop for prop in current_props
                        if prop.lower() in {p.lower() for p in previous_props}
                    ],
                    "identity_lock_targets": sorted(
                        set(previous_chars).intersection(characters_present)
                    ),
                })

            if characters_present:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "characters",
                    "question": f"Does panel {panel.panel_id} show {', '.join(characters_present)}?",
                    "expected_terms": characters_present,
                })
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "count",
                    "question": f"Does panel {panel.panel_id} show exactly {len(characters_present)} character(s)?",
                    "expected_terms": [str(len(characters_present))],
                })
            if panel.setting:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "setting",
                    "question": f"Does panel {panel.panel_id} stay in the expected setting?",
                    "expected_terms": self._split_state_terms(panel.setting)[:3] or [panel.setting],
                })
            if current_props:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "props",
                    "question": f"Are the key props visible in panel {panel.panel_id}?",
                    "expected_terms": current_props[:3],
                })
            if action_beats:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "action",
                    "question": f"Does panel {panel.panel_id} depict the expected action?",
                    "expected_terms": action_beats[:2],
                })
            if emotion_cues:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "expression",
                    "question": f"Does panel {panel.panel_id} show the expected facial expression?",
                    "expected_terms": sorted(set(emotion_cues.values())),
                })
            for entity in panel_entities:
                if entity.wardrobe_terms:
                    story_questions.append({
                        "panel_id": panel.panel_id,
                        "category": "wardrobe",
                        "question": f"Does panel {panel.panel_id} keep {entity.name}'s clothing/color identity?",
                        "expected_terms": entity.wardrobe_terms[:2] + entity.wardrobe_palette[:1],
                    })
                if entity.face_terms:
                    story_questions.append({
                        "panel_id": panel.panel_id,
                        "category": "identity_detail",
                        "question": f"Does panel {panel.panel_id} preserve {entity.name}'s face details?",
                        "expected_terms": entity.face_terms[:2],
                    })
            if previous_panel is not None and continuity_from_prev:
                story_questions.append({
                    "panel_id": panel.panel_id,
                    "category": "continuity",
                    "question": f"Does panel {panel.panel_id} preserve continuity from panel {previous_panel.panel_id}?",
                    "expected_terms": continuity_from_prev[:2],
                })

            previous_panel = panel
            previous_props = current_props
            previous_chars = characters_present

        for char_name, counts in emotion_counts.items():
            if not counts or char_name not in character_states:
                continue
            dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            character_states[char_name].emotion_state = dominant

        narrative_goals = []
        if panels:
            first_actions = self._infer_action_beats(panels[0])
            last_actions = self._infer_action_beats(panels[-1])
            if first_actions:
                narrative_goals.append(f"establish {first_actions[0]}")
            if last_actions:
                narrative_goals.append(f"resolve with {last_actions[0]}")
        narrative_goals.append(f"narrative_arc={narrative_arc}")

        global_constraints = list(dict.fromkeys([constraint for constraint in consistency_constraints if constraint]))
        if time_anchor:
            global_constraints.append(f"time_anchor={time_anchor}")
        if location_graph:
            global_constraints.append(f"location_anchor={location_graph[0]}")
        global_constraints = list(dict.fromkeys(global_constraints))

        return StoryState(
            story_id=story_id,
            theme_style=global_style,
            time_anchor=time_anchor,
            global_location_graph=location_graph,
            global_constraints=global_constraints,
            narrative_goals=narrative_goals,
            character_states=character_states,
            persistent_props=persistent_props,
            scene_slots=location_graph,
            stateful_objects=sorted(stateful_objects),
            panel_states=panel_states,
            cross_panel_links=cross_panel_links,
            story_questions_for_eval=story_questions,
        )

    def _critique_story_state(self, story_state: StoryState) -> List[Dict[str, Any]]:
        """Generate machine-readable state issues for a later patch pass."""
        issues = []
        multi_character_story = len(story_state.character_states) >= 2

        for panel_state in story_state.panel_states:
            directional_tokens = ("left", "right", "front", "behind", "next to", "beside", "facing")
            weak_layout = bool(panel_state.spatial_layout) and not any(
                token in panel_state.spatial_layout.lower() for token in directional_tokens
            )
            if multi_character_story and len(panel_state.characters_present) >= 2 and (not panel_state.spatial_layout or weak_layout):
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_spatial_layout" if not panel_state.spatial_layout else "weak_spatial_layout",
                    "severity": "medium",
                    "details": "Multi-character panel lacks explicit left/right/front/back layout.",
                })

            if panel_state.panel_id > 1 and not panel_state.continuity_from_prev:
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_continuity_link",
                    "severity": "medium",
                    "details": "Panel has no explicit continuity edge from previous panel.",
                })

            if not panel_state.must_show:
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_must_show",
                    "severity": "high",
                    "details": "Panel lacks explicit must_show contract for render/eval.",
                })

            if not panel_state.action_beats:
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_action_beats",
                    "severity": "medium",
                    "details": "Panel lacks explicit action beats.",
                })
            if panel_state.expected_count <= 0:
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_expected_count",
                    "severity": "high",
                    "details": "Panel lacks an explicit entity count contract.",
                })
            if panel_state.characters_present and not panel_state.panel_entities:
                issues.append({
                    "panel_id": panel_state.panel_id,
                    "issue_type": "missing_panel_entities",
                    "severity": "high",
                    "details": "Visible characters are not backed by per-entity state blocks.",
                })

        if not story_state.story_questions_for_eval:
            issues.append({
                "panel_id": None,
                "issue_type": "missing_eval_questions",
                "severity": "high",
                "details": "StoryState has no state-grounded eval questions.",
            })

        return issues

    def _patch_story_state(
        self,
        story_state: StoryState,
        issues: List[Dict[str, Any]]
    ) -> StoryState:
        """Apply a deterministic patch pass to repair generic state gaps."""
        actions = []
        panel_map = {panel_state.panel_id: panel_state for panel_state in story_state.panel_states}

        for issue in issues:
            panel_id = issue.get("panel_id")
            issue_type = issue.get("issue_type")
            panel_state = panel_map.get(panel_id) if panel_id is not None else None

            if issue_type in {"missing_spatial_layout", "weak_spatial_layout"} and panel_state is not None:
                previous_layout = panel_state.spatial_layout
                panel_state.spatial_layout = "characters clearly separated with stable left/right staging"
                if previous_layout:
                    panel_state.must_show = [
                        item for item in panel_state.must_show
                        if str(item).strip().lower() != previous_layout.strip().lower()
                    ]
                panel_state.must_show.append(panel_state.spatial_layout)
                actions.append({
                    "panel_id": panel_id,
                    "action": "added_spatial_layout_anchor",
                    "value": panel_state.spatial_layout,
                })

            elif issue_type == "missing_continuity_link" and panel_state is not None:
                continuity_text = "preserve visible setting, props, and character identity from previous panel"
                panel_state.continuity_from_prev.append(continuity_text)
                if continuity_text not in panel_state.must_show:
                    panel_state.must_show.append(continuity_text)
                actions.append({
                    "panel_id": panel_id,
                    "action": "added_generic_continuity_anchor",
                    "value": continuity_text,
                })

            elif issue_type == "missing_must_show" and panel_state is not None:
                fallback_targets = panel_state.characters_present[:]
                if panel_state.action_beats:
                    fallback_targets.append(panel_state.action_beats[0])
                panel_state.must_show = list(dict.fromkeys(fallback_targets))
                actions.append({
                    "panel_id": panel_id,
                    "action": "reconstructed_must_show",
                    "value": panel_state.must_show,
                })

            elif issue_type == "missing_action_beats" and panel_state is not None:
                actions.append({
                    "panel_id": panel_id,
                    "action": "left_action_beats_empty_for_scene_driven_panel",
                    "value": [],
                })
            elif issue_type == "missing_expected_count" and panel_state is not None:
                panel_state.expected_count = len(panel_state.characters_present)
                actions.append({
                    "panel_id": panel_id,
                    "action": "reconstructed_expected_count",
                    "value": panel_state.expected_count,
                })
            elif issue_type == "missing_panel_entities" and panel_state is not None:
                panel_state.panel_entities = [
                    PanelEntityState(name=char_name)
                    for char_name in panel_state.characters_present
                ]
                actions.append({
                    "panel_id": panel_id,
                    "action": "reconstructed_panel_entities",
                    "value": [entity.name for entity in panel_state.panel_entities],
                })

            elif issue_type == "missing_eval_questions":
                for panel_state in story_state.panel_states:
                    story_state.story_questions_for_eval.append({
                        "panel_id": panel_state.panel_id,
                        "category": "must_show",
                        "question": f"Does panel {panel_state.panel_id} satisfy its must_show contract?",
                        "expected_terms": panel_state.must_show[:3],
                    })
                actions.append({
                    "panel_id": None,
                    "action": "generated_fallback_eval_questions",
                    "value": len(story_state.story_questions_for_eval),
                })

        story_state.refinement_actions = actions
        story_state.refinement_issues = self._critique_story_state(story_state)
        return story_state

    def parse_llm_response(self, llm_output: str, raw_text: str = "", scenes: List = None) -> ProductionBoard:
        """
        Parse LLM output to build ProductionBoard object

        Args:
            llm_output: JSON string from LLM
            raw_text: Original raw text for fallback
            scenes: List of parsed scenes from initial parsing

        Returns:
            ProductionBoard: Structured production blueprint
        """
        if scenes is None:
            scenes = []

        def normalize_text_field(value: Any, default: str = "") -> str:
            if value is None:
                return default
            if isinstance(value, list):
                flattened = [str(item).strip() for item in value if str(item).strip()]
                return ", ".join(flattened) if flattened else default
            return str(value).strip()
        
        # Clean potential markdown markers
        cleaned = llm_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'```$', '', cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[Parser] Warning: Failed to parse LLM JSON ({e}), using rule-based fallback")
            fallback_data = json.loads(self._rule_based_parse({"characters": [], "scenes": [], "raw_text": raw_text}))
            data = fallback_data

        # Build character dictionary
        # Handle both dict format and list format from LLM
        # Also use original parsed character names for proper matching
        characters = {}
        char_list = data.get("characters", {})
        
        # Get original character names from scenes for proper matching
        original_names = []
        if scenes:
            for scene in scenes:
                content = scene.get("content", "")
                found = re.findall(r'<([^>]+)>', content)
                original_names.extend(found)
            original_names = list(dict.fromkeys(original_names))  # Remove duplicates, preserve order
        
        if isinstance(char_list, list):
            # LLM returned characters as a list
            # Match LLM characters to original names by trying to find matching tokens
            used_original_names = []
            used_original_lower = set()
            
            for idx, char_data in enumerate(char_list):
                if isinstance(char_data, dict):
                    name = char_data.get("name", "")
                    
                    if name:
                        # LLM provided a name - try to match case with original
                        name_lower = name.lower()
                        matched_orig = None
                        for orig_name in original_names:
                            if orig_name.lower() == name_lower:
                                matched_orig = orig_name
                                used_original_names.append(orig_name)
                                used_original_lower.add(orig_name.lower())
                                break
                        
                        # Use original name if found, otherwise keep LLM's name
                        if matched_orig:
                            name = matched_orig
                    else:
                        # No name provided, try to extract from token
                        token = char_data.get("token", "")
                        # Handle "sks Name" or "sks_name" format (case-insensitive)
                        if "sks" in token.lower():
                            match = re.search(r'sks[\s_]*(.+)', token, re.IGNORECASE)
                            if match:
                                token_name = match.group(1).strip()
                                # Try to match with original names (case-insensitive)
                                for orig_name in original_names:
                                    if orig_name.lower() == token_name.lower() and orig_name not in used_original_names:
                                        name = orig_name
                                        used_original_names.append(orig_name)
                                        break
                                if not name:
                                    # Use token name capitalized
                                    name = token_name.title()
                    
                    # If still no name, try to use original names in order
                    # Use case-insensitive comparison to avoid duplicates
                    if not name and idx < len(original_names):
                        used_original_lower = {n.lower() for n in used_original_names}
                        for orig_name in original_names:
                            if orig_name.lower() not in used_original_lower:
                                name = orig_name
                                used_original_names.append(orig_name)
                                used_original_lower.add(orig_name.lower())
                                break
                    
                    if name:
                        characters[name] = Character(
                            name=name,
                            visual_description=char_data.get("visual_description", ""),
                            token=char_data.get("token", f"sks {name}").replace("_", " "),  # FIX: Replace underscore with space
                            key_attributes=char_data.get("key_attributes", []),
                            clothing=char_data.get("clothing", ""),
                            appearance_details=char_data.get("appearance_details", ""),
                            entity_type=char_data.get("entity_type", "human"),
                            gender=char_data.get("gender", ""),
                            age_bucket=char_data.get("age_bucket", char_data.get("age_category", "")),
                            wardrobe_palette=char_data.get("wardrobe_palette", []),
                            face_traits=char_data.get("face_traits", []),
                        )
        else:
            # LLM returned characters as a dictionary
            # Keys might be token names like "sks Jack" - need to normalize
            used_original_names = []
            used_original_lower = set()
            
            for raw_name, char_data in char_list.items():
                if isinstance(char_data, dict):
                    name = raw_name
                    
                    # If the key is a token like "sks Jack", try to match with original names
                    if "sks" in raw_name.lower():
                        # Extract name from token
                        match = re.search(r'sks[\s_]*(.+)', raw_name, re.IGNORECASE)
                        if match:
                            token_name = match.group(1).strip()
                            # Try to match with original names (case-insensitive)
                            for orig_name in original_names:
                                if orig_name.lower() == token_name.lower() and orig_name.lower() not in used_original_lower:
                                    name = orig_name
                                    used_original_names.append(orig_name)
                                    used_original_lower.add(orig_name.lower())
                                    break
                            if name == raw_name:  # No match found
                                name = token_name.title()
                    
                    characters[name] = Character(
                        name=name,
                        visual_description=char_data.get("visual_description", ""),
                        token=char_data.get("token", f"sks {name}").replace("_", " "),  # FIX: Replace underscore with space
                        key_attributes=char_data.get("key_attributes", []),
                        clothing=char_data.get("clothing", ""),
                        appearance_details=char_data.get("appearance_details", ""),
                        entity_type=char_data.get("entity_type", "human"),
                        gender=char_data.get("gender", ""),
                        age_bucket=char_data.get("age_bucket", char_data.get("age_category", "")),
                        wardrobe_palette=char_data.get("wardrobe_palette", []),
                        face_traits=char_data.get("face_traits", []),
                    )
        
        # CRITICAL FIX: Check for MISSING characters that are in original_names but not in LLM output
        # LLM sometimes only returns some characters (e.g., Jack but not Sara in "<Jack> and <Sara>")
        # Use case-insensitive comparison to avoid duplicates like "milo" vs "Milo"
        existing_names_lower = {k.lower() for k in characters.keys()}
        missing_chars = [name for name in original_names if name.lower() not in existing_names_lower]
        
        if missing_chars:
            print(f"[Director] Warning: LLM missed characters: {missing_chars}. Creating placeholders.")
            for name in missing_chars:
                characters[name] = Character(
                    name=name,
                    visual_description="",  # Will be filled by the character enhancement logic below
                    token=f"sks {name.lower().replace(' ', '_')}",
                    key_attributes=[],
                    clothing="",
                    appearance_details="",
                )
        
        # Also handle case where LLM returned empty characters dict entirely
        if not characters and original_names:
            print(f"[Director] Warning: LLM returned empty characters. Creating from script: {original_names}")
            for name in original_names:
                characters[name] = Character(
                    name=name,
                    visual_description="",
                    token=f"sks {name.lower().replace(' ', '_')}",
                    key_attributes=[],
                    clothing="",
                    appearance_details="",
                )
        
        # CRITICAL FIX: Remove duplicate characters caused by case differences
        # If LLM returns both "milo" and "Milo", keep the one whose visual_description is actually USED in enhanced_prompts
        if len(characters) > len(set(k.lower() for k in characters.keys())):
            print(f"[Director] Warning: Found duplicate characters with different case. Deduplicating...")
            
            # Analyze panels to see which character's visual_description is actually USED in enhanced_prompts
            panels_data = data.get("panels", [])
            
            # Count based on visual_description matching in enhanced_prompts
            desc_usage = {}
            for key in characters.keys():
                char_info = characters[key]
                visual_desc = char_info.visual_description.lower() if char_info.visual_description else ""
                desc_usage[key] = 0
                
                for panel in panels_data:
                    ep = (panel.get("enhanced_prompt", "") or "").lower()
                    # Check if visual_description (first 50 chars to avoid full match issues) appears in enhanced_prompt
                    if visual_desc and len(visual_desc) > 20:
                        # Use first 40 chars of visual_description for matching
                        desc_prefix = visual_desc[:40]
                        if desc_prefix in ep:
                            desc_usage[key] += 1
            
            # Group keys by lowercase
            groups = {}
            for key in characters.keys():
                key_lower = key.lower()
                if key_lower not in groups:
                    groups[key_lower] = []
                groups[key_lower].append(key)
            
            # For each group, determine which to keep
            keys_to_remove = []
            for key_lower, keys in groups.items():
                if len(keys) > 1:
                    # Multiple versions of the same name
                    # PRIORITY: 
                    # 1. Use the one whose visual_description appears in enhanced_prompts
                    # 2. OR match original_names exactly
                    # 3. Or default to first
                    
                    keep_key = None
                    
                    # Find the one with most visual_description usage in enhanced_prompts
                    max_usage = -1
                    for key in keys:
                        usage = desc_usage.get(key, 0)
                        if usage > max_usage:
                            max_usage = usage
                            keep_key = key
                    
                    # If there's a clear winner by description usage, use it
                    if max_usage > 0:
                        # Found a clear winner by description usage
                        pass
                    else:
                        # No description usage - prefer original_names match
                        for key in keys:
                            if key in original_names:
                                keep_key = key
                                break
                    
                    if not keep_key:
                        keep_key = keys[0]  # Default to first
                    
                    print(f"  Keeping: {keep_key} (desc usage: {desc_usage.get(keep_key, 0)}, removing: {[k for k in keys if k != keep_key]})")
                    for key in keys:
                        if key != keep_key:
                            print(f"    Removing duplicate: {key}")
                            keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del characters[key]

        for char_name, char_info in characters.items():
            self._normalize_character_entity(char_name, char_info, raw_text)

        # Build panel list - need to map LLM output to actual scene content
        panels = []
        llm_panels = data.get("panels", [])
        num_scenes = len(scenes)
        num_llm_panels = len(llm_panels)
        
        # CRITICAL FIX: Ensure panel count matches scene count
        # If LLM output fewer panels than scenes, we need to fill in the gaps
        if num_llm_panels < num_scenes:
            print(f"[Director] Warning: LLM output {num_llm_panels} panels but script has {num_scenes} scenes. Filling gaps...")
        
        for idx in range(num_scenes):  # Always iterate over all scenes
            panel_data = llm_panels[idx] if idx < num_llm_panels else None
            
            # Get raw_prompt from actual scene content
            raw_prompt = scenes[idx].get('content', f"Scene {idx + 1}") if idx < len(scenes) else f"Scene {idx + 1}"
            
            # CRITICAL FIX: Ensure enhanced_prompt includes character description
            enhanced_prompt = ""
            shot_type = "medium"
            time_of_day = "daytime"
            setting = ""
            key_objects = ""
            
            if panel_data and isinstance(panel_data, dict):
                enhanced_prompt = normalize_text_field(panel_data.get('enhanced_prompt', ''))
                shot_type = normalize_text_field(panel_data.get('shot_type', 'medium'), 'medium')
                time_of_day = normalize_text_field(panel_data.get('time_of_day', 'daytime'), 'daytime')
                setting = normalize_text_field(panel_data.get('setting', ''))
                key_objects = normalize_text_field(panel_data.get('key_objects', ''))
                
                # CRITICAL FIX: If enhanced_prompt is too short or missing character, add character info
                if not enhanced_prompt or len(enhanced_prompt) < 30:
                    # Try to get from key_actions or construct
                    actions = panel_data.get('key_actions', [])
                    if actions:
                        enhanced_prompt = actions[0] if isinstance(actions[0], str) else str(actions[0])
                
                # If still empty, use raw_prompt
                if not enhanced_prompt:
                    enhanced_prompt = raw_prompt
                
                # CRITICAL FIX: Ensure character name appears in enhanced_prompt
                # First, try to extract character name from raw_prompt
                char_in_scene = self._extract_char_from_raw_prompt(raw_prompt, list(characters.keys()))
                
                # If no character in raw_prompt (environmental description like "city lights come on"),
                # use the first character from previous panels as the main character
                if not char_in_scene and characters and idx > 0:
                    # Get first character's name as default
                    first_char = list(characters.keys())[0]
                    char_in_scene = first_char
                
                if char_in_scene and char_in_scene not in enhanced_prompt[:50]:
                    # Prepend character name to enhanced_prompt
                    char_info = characters.get(char_in_scene)
                    if char_info and char_info.visual_description and "person" not in char_info.visual_description.lower():
                        enhanced_prompt = f"{char_in_scene}, {char_info.visual_description}, {enhanced_prompt}"
                    else:
                        enhanced_prompt = f"{char_in_scene}, {enhanced_prompt}"
            else:
                # Panel data missing - use raw_prompt
                char_in_scene = self._extract_char_from_raw_prompt(raw_prompt, list(characters.keys()))
                if char_in_scene:
                    char_info = characters.get(char_in_scene)
                    if char_info:
                        enhanced_prompt = f"{char_in_scene}, {char_info.visual_description}, {raw_prompt}"
                    else:
                        enhanced_prompt = f"{char_in_scene}, {raw_prompt}"
                else:
                    enhanced_prompt = raw_prompt
            
            # CRITICAL FIX: Infer better shot type based on panel content
            # - Environmental descriptions (lights, scenery) should NOT be close-ups
            # - Panels with pronouns and no named character might be reactions
            raw_lower = raw_prompt.lower()
            
            # Environmental/description-only panels should be wide or establishing
            env_keywords = ['lights', 'skyline', 'scenery', 'view', 'landscape', 'come on', 'begins', 'starts']
            is_env_panel = any(kw in raw_lower for kw in env_keywords) and not any(c.lower() in raw_lower for c in characters.keys())
            
            # If shot type is close-up/medium but panel is environmental, adjust to wide
            if is_env_panel and shot_type in ['closeup', 'medium']:
                shot_type = "wide"
            
            # If panel mentions "looks" or "smiles" or "reacts", use medium shot to show character
            reaction_keywords = ['looks', 'smile', 'laugh', 'wave', 'pause', 'stand', 'sit', 'walk']
            if any(kw in raw_lower for kw in reaction_keywords) and shot_type in ['wide', 'extreme_closeup']:
                shot_type = "medium"
            
            # CRITICAL FIX: If setting is empty but this is a continuation panel,
            # use the previous panel's setting for consistency
            if not setting and idx > 0 and panels:
                setting = panels[-1].setting if hasattr(panels[-1], 'setting') else ""
            
            panel_data_with_defaults = {
                'panel_id': idx + 1,
                'raw_prompt': normalize_text_field(raw_prompt),
                'enhanced_prompt': normalize_text_field(enhanced_prompt, normalize_text_field(raw_prompt)),
                'shot_type': normalize_text_field(shot_type, 'medium'),
                'camera_movement': normalize_text_field(panel_data.get('camera_movement', 'static') if panel_data else 'static', 'static'),
                'lighting_mood': normalize_text_field(panel_data.get('lighting_mood', 'natural') if panel_data else 'natural', 'natural'),
                'key_actions': [str(item).strip() for item in (panel_data.get('key_actions', []) if panel_data else []) if str(item).strip()],
                'interactions': panel_data.get('interactions', []) if panel_data else [],
                'setting': normalize_text_field(setting),
                'time_of_day': normalize_text_field(time_of_day, 'daytime'),
                'weather': normalize_text_field(panel_data.get('weather', 'clear') if panel_data else 'clear', 'clear'),
                'key_objects': normalize_text_field(key_objects)
            }
            panels.append(Panel(**panel_data_with_defaults))

        # CRITICAL FIX: Ensure all characters have complete descriptions
        # If any character has empty visual_description, key_attributes, etc., fill them in
        import random
        
        # Detect character type: animal, robot, or human
        animal_keywords = ['dog', 'cat', 'bird', 'rabbit', 'horse', 'lion', 'tiger', 'bear', 
                         'wolf', 'fox', 'deer', 'elephant', 'monkey', 'panda', 'koala',
                         'fish', 'owl', 'eagle', 'shark', 'duck', 'chicken', 'pig', 'cow',
                         'sheep', 'goat', 'snake', 'lizard', 'turtle', 'frog', 'puppy', 'kitten']
        robot_keywords = ['robot', 'android', 'mecha', 'drone', 'automaton', 'cyborg',
                         'machine', 'droid', 'golem', 'mechanoid', 'gynoid']
        
        def _detect_character_type(name: str) -> str:
            nl = name.lower()
            if any(kw in nl for kw in robot_keywords):
                return "robot"
            if any(kw in nl for kw in animal_keywords):
                return "animal"
            return "human"
        
        for char_name, char_info in characters.items():
            random.seed(self._stable_seed(char_name))
            char_type = _detect_character_type(char_name)
            char_info.entity_type = self._detect_entity_type(
                char_name,
                char_info.visual_description,
                char_info.appearance_details,
            )
            
            if char_type == "robot":
                robot_body_types = ["humanoid metallic body", "angular mechanical chassis",
                                    "sleek robotic frame", "boxy industrial casing"]
                robot_materials = ["brushed steel", "matte black metal", "white enamel plating",
                                   "brass and copper", "carbon fiber composite", "chrome finish"]
                robot_features = ["glowing LED indicators", "visible mechanical joints",
                                  "articulated limbs", "optical sensor array",
                                  "circuit patterns on surface", "telescoping appendages"]
                robot_colors = ["silver", "matte black", "white", "gunmetal gray",
                                "copper", "brushed aluminum"]
                
                body = random.choice(robot_body_types)
                material = random.choice(robot_materials)
                feature = random.choice(robot_features)
                color = random.choice(robot_colors)
                
                char_info.visual_description = (
                    f"A {color} robot with {body}, {material} construction, "
                    f"{feature}, realistic mechanical details"
                )
                char_info.key_attributes = [color, body, material, feature, "robot", "mechanical"]
                char_info.clothing = ""
                char_info.appearance_details = f"robot with {color} {material} {feature}"
                
            elif char_type == "animal":
                animal_type = char_name
                for kw in animal_keywords:
                    if kw in char_name.lower():
                        animal_type = kw
                        break
                
                # Per-taxon features (mammal vs bird vs reptile vs fish)
                bird_keywords = ['bird', 'owl', 'eagle', 'duck', 'chicken', 'penguin', 'parrot']
                reptile_keywords = ['snake', 'lizard', 'turtle', 'frog', 'gecko', 'chameleon']
                fish_keywords = ['fish', 'shark', 'dolphin', 'whale', 'goldfish']
                
                is_bird = any(kw in animal_type.lower() for kw in bird_keywords)
                is_reptile = any(kw in animal_type.lower() for kw in reptile_keywords)
                is_fish = any(kw in animal_type.lower() for kw in fish_keywords)
                
                if is_bird:
                    features_pool = ["smooth feathers", "glossy plumage", "soft downy feathers",
                                     "sleek wings", "bright plumage", "intricate feather patterns"]
                    feature_label = "feathers, realistic bird anatomy"
                elif is_reptile:
                    features_pool = ["scaly skin", "patterned scales", "smooth cool skin",
                                     "leathery hide", "overlapping scales"]
                    feature_label = "scales, realistic reptile proportions"
                elif is_fish:
                    features_pool = ["shiny scales", "smooth skin", "iridescent coloring",
                                     "streamlined body", "translucent fins"]
                    feature_label = "scales, realistic fish anatomy"
                else:
                    features_pool = ["fluffy fur", "shiny coat", "soft fur", "thick fur",
                                     "glossy pelt", "dense woolly coat"]
                    feature_label = "fur, realistic mammal proportions"
                
                colors = ["golden brown", "black and white", "brown", "gray", "white",
                         "orange and white", "black", "cream colored", "rust colored",
                         "spotted brown", "striped gray", "mottled tan"]
                expressions = ["alert expression", "curious gaze", "friendly eyes",
                              "happy demeanor", "calm presence", "playful stance",
                              "watchful posture", "gentle manner"]
                
                color = random.choice(colors)
                feature = random.choice(features_pool)
                expression = random.choice(expressions)
                
                char_info.visual_description = (
                    f"A {color} {animal_type} with {feature}, {expression}, {feature_label}"
                )
                char_info.key_attributes = [color, feature, expression, f"realistic {animal_type}"]
                char_info.clothing = ""
                char_info.appearance_details = f"{animal_type} with {color} {feature}"
                
            else:
                # Human character - use existing logic
                # Infer gender and age from name
                gender = self._infer_gender_bucket(char_info)
                if gender == "neutral":
                    gender = self._infer_gender_fallback(char_name)
                age_bucket = self._infer_age_bucket(char_info)
                age_map = {
                    "baby": "baby",
                    "toddler": "toddler",
                    "child": "child",
                    "teen": "teen",
                    "young_adult": "young adult",
                    "adult": "adult",
                    "elderly": "elderly adult",
                }
                age = age_map.get(age_bucket, "young adult")
                childlike = age_bucket in {"baby", "toddler", "child"}
                subject_descriptor = age if childlike or gender not in {"male", "female"} else f"{age} {gender}"
                
                hair_colors = ["black hair", "brown hair", "blonde hair", "dark brown hair", "auburn hair", "red hair"]
                hair_styles = ["short hair", "medium-length hair", "long hair", "wavy hair", "straight hair", "messy hair"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes", "gray eyes"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tan skin"]
                builds = ["slim", "average", "athletic", "medium build"]
                
                # Check if visual_description is too generic or contradictory
                is_generic = "person" in char_info.visual_description.lower() if char_info.visual_description else True
                
                if not char_info.visual_description or is_generic:
                    # Generate specific visual description with actual features
                    hair = random.choice(hair_colors)
                    style = random.choice(hair_styles)
                    eyes = random.choice(eye_colors)
                    skin = random.choice(skin_tones)
                    build = random.choice(builds)
                    
                    if childlike:
                        clothing_options = [
                            "colorful children's clothes",
                            "soft play clothes",
                            "bright t-shirt and shorts",
                            "cozy kid outfit",
                        ]
                    elif gender == "female":
                        clothing_options = [
                            "blue blouse and jeans",
                            "casual red sweater and black skirt",
                            "red dress",
                            "green top and white pants",
                            "yellow shirt and black jeans"
                        ]
                    else:
                        clothing_options = [
                            "blue shirt and jeans",
                            "casual t-shirt and pants",
                            "gray sweater and dark jeans",
                            "green jacket and khaki pants",
                            "white shirt and black pants"
                        ]
                    
                    clothing = random.choice(clothing_options)
                    
                    # Safeguard: ensure clothing has at least one color word
                    _known_colors = {"red", "blue", "green", "yellow", "black", "white", "gray", "navy", "brown", "pink", "purple", "orange", "beige", "tan", "cream"}
                    if not any(c in clothing.lower() for c in _known_colors):
                        import random as _rnd
                        _default_color = _rnd.choice(["blue", "red", "green", "black", "navy"])
                        clothing = f"{_default_color} {clothing}"
                    
                    char_info.visual_description = (
                        f"A {subject_descriptor} with {hair}, {style}, {eyes}, {build} build, {skin}, wearing {clothing}"
                    )
                
                # CRITICAL FIX: Synchronize key_attributes with visual_description
                # If they contradict, trust visual_description as source of truth
                
                # First, fix visual_description if it's too vague
                vague_terms = ["casual outfit", "suitable for", "commute", "everyday", "typical", "simple"]
                is_vague = any(term in char_info.visual_description.lower() for term in vague_terms)
                
                if is_vague or not char_info.visual_description:
                    # Generate a more specific visual description
                    hair = random.choice(hair_colors)
                    style = random.choice(hair_styles)
                    eyes = random.choice(eye_colors)
                    skin = random.choice(skin_tones)
                    build = random.choice(builds)
                    
                    if childlike:
                        clothing = random.choice([
                            "colorful children's clothes",
                            "soft play clothes",
                            "bright t-shirt and shorts",
                            "cozy kid outfit",
                        ])
                    elif gender == "female":
                        clothing = random.choice([
                            "blue blouse and dark jeans",
                            "casual white sweater and navy pants",
                            "red floral dress",
                            "green casual top and black skirt",
                            "yellow cardigan and jeans"
                        ])
                    else:
                        clothing = random.choice([
                            "blue button-up shirt and jeans",
                            "casual gray hoodie and dark pants",
                            "green casual jacket and khaki pants",
                            "white polo shirt and navy shorts",
                            "black t-shirt and jeans"
                        ])
                    
                    char_info.visual_description = (
                        f"A {subject_descriptor} with {hair}, {style}, {eyes}, {build} build, {skin}, wearing {clothing}"
                    )
                
                # Now extract key_attributes from the fixed visual_description
                # CRITICAL: Extract CLEAN feature phrases, not full sentences
                features = char_info.visual_description.split(",")
                
                hair_from_vd = None
                eyes_from_vd = None
                clothing_from_vd = None
                
                for f in features:
                    f_lower = f.lower().strip()
                    f_clean = f.strip()
                    
                    # Hair: look for specific hair descriptions
                    if 'hair' in f_lower and not hair_from_vd:
                        # Skip if it's the whole sentence (contains "is a" or starts with article)
                        if not any(v in f_lower for v in ['is a', 'with a', 'person', 'woman', 'man', 'boy', 'girl']):
                            hair_from_vd = f_clean
                        elif 'hair' in f_lower:
                            # Extract just the hair part
                            parts = f_clean.split()
                            hair_idx = None
                            for idx, w in enumerate(parts):
                                if 'hair' in w.lower():
                                    hair_idx = idx
                                    break
                            if hair_idx and hair_idx > 0:
                                # Get 2-3 words before hair
                                start = max(0, hair_idx - 2)
                                hair_from_vd = ' '.join(parts[start:hair_idx+1])
                    
                    # Eyes
                    if 'eyes' in f_lower and not eyes_from_vd:
                        if not any(v in f_lower for v in ['is a', 'with a']):
                            eyes_from_vd = f_clean
                    
                    # Clothing
                    if 'wearing' in f_lower and not clothing_from_vd:
                        clothing_from_vd = f_clean.replace('wearing ', '')
                
                # Build clean key_attributes
                new_attrs = []
                if hair_from_vd:
                    new_attrs.append(hair_from_vd)
                if eyes_from_vd:
                    new_attrs.append(eyes_from_vd)
                if clothing_from_vd:
                    new_attrs.append(clothing_from_vd)
                new_attrs.append("realistic proportions")
                
                char_info.key_attributes = new_attrs
                
                # Update clothing field
                if clothing_from_vd:
                    char_info.clothing = clothing_from_vd
                
                # If key_attributes is still empty or too few, generate specific attributes
                if not char_info.key_attributes or len(char_info.key_attributes) <= 2:
                    hair = hair_colors[random.randint(0, len(hair_colors)-1)]
                    style = hair_styles[random.randint(0, len(hair_styles)-1)]
                    eyes = eye_colors[random.randint(0, len(eye_colors)-1)]
                    char_info.key_attributes = [f"{hair}, {style}", eyes, "realistic proportions"]
                
                # If clothing is still empty or generic
                if not char_info.clothing or "clothing" in char_info.clothing.lower():
                    char_info.clothing = "casual comfortable clothing"
                
                # If appearance_details is empty, extract from visual_description
                if not char_info.appearance_details:
                    features = [part.strip() for part in char_info.visual_description.split(",")]
                    specific_features = [f for f in features if any(x in f.lower() for x in ["hair", "eyes", "skin", "build"])]
                    if specific_features:
                        char_info.appearance_details = ", ".join(specific_features[:3])
                    else:
                        char_info.appearance_details = f"{gender} appearance with natural features"

            self._normalize_character_entity(char_name, char_info, raw_text)

        # CRITICAL FIX: Enforce time_of_day consistency across all panels
        # Analyze all panels to determine the correct time setting
        time_keywords = {
            'night': ['night', 'darkness', 'moon', 'stars', 'evening', 'dusk', 'twilight', 'nighttime'],
            'morning': ['morning', 'sunrise', 'dawn', 'breakfast', 'early', 'a.m.'],
            'afternoon': ['afternoon', 'midday', 'noon', 'sunny day', 'daytime'],
            'evening': ['evening', 'sunset', 'dusk', 'golden hour', 'dinner']
        }
        
        # Count time_of_day occurrences
        time_counts = {}
        panel_times = []
        for panel in panels:
            tod = panel.time_of_day.lower() if panel.time_of_day else "daytime"
            panel_times.append(tod)
            time_counts[tod] = time_counts.get(tod, 0) + 1
        
        # Also analyze raw_prompts and enhanced_prompts for time hints
        for panel in panels:
            text = (panel.raw_prompt + " " + panel.enhanced_prompt).lower()
            for time_cat, keywords in time_keywords.items():
                if any(kw in text for kw in keywords):
                    # Boost count for this time category
                    time_counts[time_cat] = time_counts.get(time_cat, 0) + 0.5
        
        # Find the most common/indicated time
        if time_counts:
            dominant_time = max(time_counts, key=time_counts.get)
        else:
            dominant_time = "daytime"
        
        # Fix any panels that don't match the dominant time
        corrected_panels = []
        inconsistent_count = 0
        for panel in panels:
            panel_tod = panel.time_of_day.lower() if panel.time_of_day else "daytime"
            
            # Check if this panel's time is consistent with dominant time
            is_consistent = False
            
            # Direct match
            if panel_tod == dominant_time:
                is_consistent = True
            # Check if panel mentions dominant time keywords
            panel_text = (panel.raw_prompt + " " + panel.enhanced_prompt).lower()
            dominant_keywords = time_keywords.get(dominant_time, [])
            if any(kw in panel_text for kw in dominant_keywords):
                is_consistent = True
            # Check if panel time category matches dominant time category
            for time_cat, keywords in time_keywords.items():
                if panel_tod == time_cat and any(kw in dominant_time for kw in [time_cat]):
                    is_consistent = True
                    break
            
            if not is_consistent and dominant_time not in ['daytime', 'afternoon']:
                # Only correct obvious mismatches (e.g., night story shouldn't have daytime panels)
                if dominant_time in ['night', 'evening'] and panel_tod in ['daytime', 'afternoon', 'morning']:
                    panel.time_of_day = dominant_time
                    inconsistent_count += 1
                elif dominant_time == 'morning' and panel_tod in ['night', 'evening']:
                    panel.time_of_day = dominant_time
                    inconsistent_count += 1
        
        if inconsistent_count > 0:
            print(f"[Director] Fixed {inconsistent_count} panels with inconsistent time_of_day (set to '{dominant_time}')")

        # Build complete ProductionBoard
        # CRITICAL: Generate default consistency_constraints if empty
        consistency_constraints = data.get("consistency_constraints", [])
        
        # If consistency_constraints is empty or None, generate defaults from character data
        if not consistency_constraints:
            consistency_constraints = []
            for char_name, char_info in characters.items():
                if hasattr(char_info, 'appearance_details') and char_info.appearance_details:
                    consistency_constraints.append(f"{char_name}'s appearance details: {char_info.appearance_details}")
                if hasattr(char_info, 'clothing') and char_info.clothing:
                    consistency_constraints.append(f"{char_name}'s clothing: {char_info.clothing}")
        
        story_id = f"story_{self._stable_seed(data.get('global_style', '')) % 10000}"
        story_state = self._build_story_state(
            story_id=story_id,
            raw_text=raw_text,
            characters=characters,
            panels=panels,
            global_style=data.get("global_style", "warm_cinematic_lifestyle"),
            consistency_constraints=consistency_constraints,
            narrative_arc=data.get("narrative_arc", "linear"),
        )
        story_state = self._patch_story_state(
            story_state,
            self._critique_story_state(story_state),
        )

        board = ProductionBoard(
            story_id=story_id,
            characters=characters,
            panels=panels,
            global_style=data.get("global_style", "warm_cinematic_lifestyle"),  # Use defined style
            consistency_constraints=consistency_constraints,
            narrative_arc=data.get("narrative_arc", "linear"),
            story_state=asdict(story_state),
        )

        return board

    def process_script_file(self, file_path: str) -> ProductionBoard:
        """
        Main entry point for processing a single script file

        Args:
            file_path: Path to script file

        Returns:
            ProductionBoard: Complete production blueprint
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            script_text = f.read().strip()

        print(f"[Director] Parsing script: {Path(file_path).name}")

        # Initial parsing
        parsed = self.parse_raw_script(script_text)
        print(f"[Director] Found {len(parsed['scenes'])} scenes, characters: {parsed['characters']}")

        # LLM analysis
        print(f"[Director] Calling {self.model_name} for deep analysis...")
        llm_output = self.call_llm_for_analysis(parsed)

        # Build structured output
        production_board = self.parse_llm_response(llm_output, script_text, parsed.get('scenes', []))

        print(f"[Director] Parsing complete! Style: {production_board.global_style}")

        return production_board

    def save_production_board(self, board: ProductionBoard, output_path: str):
        """Save ProductionBoard to JSON file"""
        output_data = {
            "story_id": board.story_id,
            "characters": {k: asdict(v) for k, v in board.characters.items()},
            "panels": [asdict(p) for p in board.panels],
            "global_style": board.global_style,
            "consistency_constraints": board.consistency_constraints,
            "narrative_arc": board.narrative_arc,
            "story_state": board.story_state,
            "render_plan": board.render_plan,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test with sample script
    parser = LLMScriptParser(llm_backend="local", model_name="llama3:70b")
    test_file = "storygen/data/TaskA/06.txt"

    try:
        board = parser.process_script_file(test_file)
        print(f"Story ID: {board.story_id}")
        print(f"Characters: {list(board.characters.keys())}")
        print(f"Panels: {len(board.panels)}")
        print(f"Global Style: {board.global_style}")
    except FileNotFoundError:
        print(f"[Test] Sample file not found: {test_file}")
