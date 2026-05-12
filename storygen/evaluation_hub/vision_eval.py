"""
Vision Evaluation Module
Multi-dimensional diagnostic evaluation for story image consistency.
Uses local VLM (Qwen2-VL-7B) for qualitative analysis.

Provides:
- Multi-dimensional diagnostic evaluation (not just scores)
- Version tracking across code changes
- Specific actionable feedback for debugging
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from storygen.utils.mirror_config import setup_china_mirrors, get_models_cache_dir
from storygen.evaluation_hub.state_eval import (
    _term_matches_prompt,
    build_state_eval_questions,
    derive_failure_tags,
    evaluate_prompt_state_alignment,
)

EVAL_VERSION = "1.0"
# Path relative to storygen/ package (not CWD) for robustness
STORYGEN_DIR = Path(__file__).parent.parent
VERSIONS_DIR = STORYGEN_DIR / "outputs/versions"
STRUCTURED_JUDGE_CATEGORIES = {"count", "setting", "props", "wardrobe"}
STRUCTURED_TAG_BY_CATEGORY = {
    "count": "vj_count_drift",
    "setting": "vj_scene_overwrite",
    "props": "vj_relational_prop_loss",
    "wardrobe": "vj_wardrobe_drift",
}


class VisionDiagnosticReport:
    """
    Structured multi-dimensional diagnostic report for a story generation.
    Records observations across 6 axes + version metadata.
    """
    
    def __init__(self, story_id: str, version_tag: str, config_snapshot: dict):
        self.story_id = story_id
        self.version_tag = version_tag
        self.timestamp = datetime.now().isoformat()
        self.config_snapshot = config_snapshot
        
        # 6 evaluation dimensions
        self.dimensions = {
            "character_consistency": {
                "score": 0.0,
                "findings": [],
                "issues": []
            },
            "scene_coherence": {
                "score": 0.0,
                "findings": [],
                "issues": []
            },
            "prompt_alignment": {
                "score": 0.0,
                "findings": [],
                "issues": []
            },
            "temporal_flow": {
                "score": 0.0,
                "findings": [],
                "issues": []
            },
            "style_uniformity": {
                "score": 0.0,
                "findings": [],
                "issues": []
            },
            "artifact_quality": {
                "score": 0.0,
                "findings": [],
                "issues": []
            }
        }
        
        self.summary = ""
        self.critical_issues = []
        self.quantitative = {}
        self.state_alignment = {}
        self.failure_tags = []
        self.vision_judge = {
            "model": "none",
            "confidence": "low",
            "unknown_rate": 1.0,
            "issue_tags": [],
            "panel_judgments": [],
            "cross_panel_judgments": [],
        }
    
    def add_finding(self, dimension: str, finding: str, is_issue: bool = False, 
                    severity: str = "info"):
        if dimension not in self.dimensions:
            return
        entry = {"text": finding, "severity": severity}
        self.dimensions[dimension]["findings"].append(entry)
        if is_issue:
            self.dimensions[dimension]["issues"].append(entry)
            if severity in ("high", "critical"):
                self.critical_issues.append(f"[{dimension}] {finding}")
    
    def set_dimension_score(self, dimension: str, score: float):
        if dimension in self.dimensions:
            self.dimensions[dimension]["score"] = max(0.0, min(5.0, score))
    
    def set_quantitative(self, clip_score: float, consistency_score: float):
        self.quantitative = {
            "clip_score": clip_score,
            "consistency_score": consistency_score,
            "overall": round(0.6 * clip_score + 0.4 * consistency_score, 4)
        }
    
    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "version_tag": self.version_tag,
            "timestamp": self.timestamp,
            "config": self.config_snapshot,
            "quantitative": self.quantitative,
            "dimensions": {
                k: {
                    "score": v["score"],
                    "finding_count": len(v["findings"]),
                    "issue_count": len(v["issues"]),
                    "findings": v["findings"],
                    "issues": v["issues"]
                }
                for k, v in self.dimensions.items()
            },
            "critical_issues": self.critical_issues,
            "summary": self.summary,
            "state_alignment": self.state_alignment,
            "failure_tags": self.failure_tags,
            "vision_judge": self.vision_judge,
        }


class VersionTracker:
    """
    Tracks evaluation versions over time for regression detection.
    Each evaluation run creates a versioned report.
    """
    
    def __init__(self, versions_dir: Path = VERSIONS_DIR):
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def save_report(self, report: VisionDiagnosticReport):
        """Save a diagnostic report to versioned directory."""
        story_dir = self.versions_dir / report.story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{report.version_tag}_{report.timestamp.replace(':', '-')}.json"
        filepath = story_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return str(filepath)
    
    def get_latest_report(self, story_id: str) -> Optional[dict]:
        """Get the most recent report for a story."""
        story_dir = self.versions_dir / story_id
        if not story_dir.exists():
            return None
        reports = sorted(story_dir.glob("*.json"))
        if not reports:
            return None
        with open(reports[-1]) as f:
            return json.load(f)
    
    def compare_versions(self, story_id: str, version_a: str, version_b: str) -> dict:
        """Compare two versions of the same story."""
        story_dir = self.versions_dir / story_id
        reports = list(story_dir.glob("*.json"))
        
        a_report = [r for r in reports if version_a in r.name]
        b_report = [r for r in reports if version_b in r.name]
        
        if not a_report or not b_report:
            return {"error": "Version not found"}
        
        with open(a_report[0]) as f:
            a = json.load(f)
        with open(b_report[0]) as f:
            b = json.load(f)
        
        comparison = {
            "story_id": story_id,
            "version_a": version_a,
            "version_b": version_b,
            "dimension_deltas": {}
        }
        
        for dim in a.get("dimensions", {}):
            a_score = a["dimensions"][dim]["score"]
            b_score = b["dimensions"][dim]["score"]
            comparison["dimension_deltas"][dim] = {
                "a": a_score, "b": b_score, "delta": round(b_score - a_score, 2)
            }
        
        # Overall change
        a_q = a.get("quantitative", {})
        b_q = b.get("quantitative", {})
        comparison["quantitative_delta"] = {
            "clip": {"a": a_q.get("clip_score"), "b": b_q.get("clip_score")},
            "consistency": {"a": a_q.get("consistency_score"), "b": b_q.get("consistency_score")}
        }
        
        return comparison
    
    def get_all_version_tags(self) -> List[str]:
        """List all unique version tags across all stories."""
        tags = set()
        for story_dir in self.versions_dir.iterdir():
            if story_dir.is_dir():
                for report_file in story_dir.glob("*.json"):
                    with open(report_file) as f:
                        data = json.load(f)
                    tags.add(data.get("version_tag", "unknown"))
        return sorted(tags)
    
    def summary_table(self) -> str:
        """Generate a markdown summary table comparing all versions."""
        tags = self.get_all_version_tags()
        if not tags:
            return "No versions tracked yet."
        
        lines = ["# Vision Evaluation Version History\n"]
        lines.append(f"| Story | {' | '.join(f'{t}' for t in tags)} |")
        lines.append(f"|{':---|' * (len(tags) + 1)}")
        
        for story_dir in sorted(self.versions_dir.iterdir()):
            if not story_dir.is_dir():
                continue
            scores = []
            for tag in tags:
                reports = list(story_dir.glob(f"{tag}_*.json"))
                if reports:
                    with open(reports[-1]) as f:
                        data = json.load(f)
                    q = data.get("quantitative", {})
                    overall = q.get("overall", "?")
                    scores.append(f"{overall:.3f}" if isinstance(overall, float) else "?")
                else:
                    scores.append("-")
            lines.append(f"| {story_dir.name} | {' | '.join(scores)} |")
        
        return "\n".join(lines)


class VisionEvaluator:
    """
    Multi-dimensional diagnostic evaluator using local VLM.
    Falls back gracefully if VLM is unavailable.
    """
    
    def __init__(self, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._model = None
        self._processor = None
        self._vlm_type = None
        self.version_tracker = VersionTracker()
        self._load_model()
    
    def _load_model(self):
        """Lazy-load VLM. Tries Qwen2-VL (if transformers supports it), then BLIP-2, then fallback."""
        if not torch.cuda.is_available():
            print("[VisionEval] No GPU available, using fallback analysis")
            return

        # Strategy 1: AutoModelForVision2Seq (works for Qwen2-VL, Qwen3-VL, LLaVA, etc.)
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
            model_paths_to_try = [
                # Qwen2-VL-7B cached via HF mirror
                str(get_models_cache_dir() / "models--Qwen--Qwen2-VL-7B-Instruct"),
                # Already cached from other users
                "/data/baiyixue/inference_model/Qwen3-VL-30B-A3B-Instruct",
            ]
            for mp in model_paths_to_try:
                if Path(mp).exists() and any(Path(mp).rglob("*.safetensors")):
                    print(f"[VisionEval] Found VLM at {mp}, loading...")
                    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                    self._model = AutoModelForVision2Seq.from_pretrained(
                        mp, quantization_config=quant, dtype=torch.float16,
                        device_map="auto", local_files_only=True, trust_remote_code=True,
                    )
                    self._processor = AutoProcessor.from_pretrained(
                        mp, local_files_only=True, trust_remote_code=True,
                    )
                    print(f"[VisionEval] VLM loaded successfully: {Path(mp).name}")
                    return
        except Exception as e:
            print(f"[VisionEval] AutoModelForVision2Seq approach failed: {e}")

        # Strategy 2: BLIP-2 (always available in transformers)
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        except Exception:
            quant = None
        setup_china_mirrors()
        blip2_candidates = ["Salesforce/blip2-opt-2.7b", "Salesforce/blip2-flan-t5-xl"]
        for model_id in blip2_candidates:
            try:
                print(f"[VisionEval] Trying BLIP-2: {model_id}...")
                load_kwargs = {
                    "device_map": {"": self.device},
                    "cache_dir": str(get_models_cache_dir()),
                }
                if quant is not None:
                    load_kwargs["quantization_config"] = quant
                    load_kwargs["dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float16
                self._model = Blip2ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
                self._processor = Blip2Processor.from_pretrained(
                    model_id, cache_dir=str(get_models_cache_dir()),
                )
                self._vlm_type = "blip2"
                print(f"[VisionEval] BLIP-2 loaded: {model_id}")
                return
            except Exception as e:
                print(f"[VisionEval] BLIP-2 {model_id} unavailable: {e}")
                if "out of memory" in str(e).lower():
                    break

        print("[VisionEval] No VLM available, using pixel-level fallback analysis")
        
    def is_available(self) -> bool:
        return self._model is not None

    def _extract_wardrobe_subject(self, question: dict[str, Any]) -> str:
        match = re.search(
            r"keep\s+(.+?)'s clothing",
            str(question.get("question", "") or ""),
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        expected_terms = question.get("expected_terms", []) or []
        return str(expected_terms[0] if expected_terms else "subject").strip()

    def _select_structured_panel_questions(self, story_state: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        selected: dict[int, list[dict[str, Any]]] = {}
        per_panel_categories: dict[int, set[str]] = {}
        per_panel_wardrobe = {}
        for question in build_state_eval_questions(story_state):
            if not isinstance(question, dict):
                continue
            category = str(question.get("category", "") or "").strip().lower()
            if category not in STRUCTURED_JUDGE_CATEGORIES:
                continue
            expected_terms = [str(term).strip() for term in (question.get("expected_terms", []) or []) if str(term).strip()]
            if not expected_terms:
                continue
            panel_id = int(question.get("panel_id", 0) or 0)
            if panel_id <= 0:
                continue
            panel_selected = selected.setdefault(panel_id, [])
            panel_categories = per_panel_categories.setdefault(panel_id, set())
            if category == "wardrobe":
                wardrobe_count = per_panel_wardrobe.get(panel_id, 0)
                if wardrobe_count >= 2:
                    continue
                subject = self._extract_wardrobe_subject(question)
                key = f"wardrobe_{re.sub(r'[^a-z0-9]+', '_', subject.lower()).strip('_') or wardrobe_count + 1}"
                panel_selected.append({
                    "key": key,
                    "category": category,
                    "question": str(question.get("question", "") or ""),
                    "expected_terms": expected_terms[:3],
                    "subject": subject,
                })
                per_panel_wardrobe[panel_id] = wardrobe_count + 1
                continue
            if category in panel_categories:
                continue
            panel_categories.add(category)
            key = category
            panel_selected.append({
                "key": key,
                "category": category,
                "question": str(question.get("question", "") or ""),
                "expected_terms": expected_terms[:3],
                "subject": "",
            })
        return selected

    def _build_structured_panel_prompt(self, panel_id: int, checks: list[dict[str, Any]]) -> str:
        lines = [
            "You are checking one storyboard panel image.",
            "Answer every item on its own line using the exact format key=value.",
            "For count use a number or unknown. For other keys use yes, no, or unknown only.",
            "Do not add extra commentary.",
            "",
            f"Panel {panel_id} checks:",
        ]
        for check in checks:
            key = check["key"]
            expected = ", ".join(check["expected_terms"][:2])
            if check["category"] == "count":
                lines.append(f"{key}=<number or unknown> :: How many main people or animals are clearly visible?")
            elif check["category"] == "setting":
                lines.append(f"{key}=<yes|no|unknown> :: Does the image show this setting: {expected}?")
            elif check["category"] == "props":
                lines.append(f"{key}=<yes|no|unknown> :: Is at least one of these props clearly visible: {expected}?")
            elif check["category"] == "wardrobe":
                lines.append(
                    f"{key}=<yes|no|unknown> :: Is there a visible character matching this clothing cue: {expected}?"
                )
        return "\n".join(lines)

    def _build_single_check_prompt(self, panel_id: int, check: dict[str, Any]) -> str:
        expected = ", ".join(check["expected_terms"][:2])
        if check["category"] == "count":
            return (
                f"You are checking panel {panel_id} of a storyboard. "
                "How many main people or animals are clearly visible in this single image? "
                "Answer with one token only: 0, 1, 2, 3, 4, or unknown."
            )
        if check["category"] == "setting":
            return (
                f"You are checking panel {panel_id} of a storyboard. "
                f"Does this single image show this setting: {expected}? "
                "Answer with one token only: yes, no, or unknown."
            )
        if check["category"] == "props":
            return (
                f"You are checking panel {panel_id} of a storyboard. "
                f"Is at least one of these props clearly visible in the single image: {expected}? "
                "Answer with one token only: yes, no, or unknown."
            )
        return (
            f"You are checking panel {panel_id} of a storyboard. "
            f"Is there a visible character matching this clothing cue: {expected}? "
            "Answer with one token only: yes, no, or unknown."
        )

    def _build_panel_caption_prompt(self, panel_id: int) -> str:
        return (
            f"Describe panel {panel_id} of this storyboard in one concise sentence. "
            "Mention how many people or animals are visible, the setting, major props, and the most obvious clothing colors or garments."
        )

    def _infer_check_from_caption(self, caption: str, check: dict[str, Any]) -> tuple[str, str]:
        caption_text = str(caption or "").strip()
        if not caption_text:
            return "unknown", "unknown"

        if check["category"] == "count":
            observed_count = self._normalize_count_answer(caption_text)
            expected_count = None
            for term in check["expected_terms"]:
                if str(term).isdigit():
                    expected_count = int(term)
                    break
            if observed_count is None or expected_count is None:
                return "unknown", "unknown"
            return str(observed_count), "pass" if observed_count == expected_count else "fail"

        matched_terms = [
            term for term in check["expected_terms"]
            if _term_matches_prompt(term, caption_text, check["category"])
        ]
        if matched_terms:
            return ", ".join(matched_terms[:2]), "pass"

        lowered = caption_text.lower()
        if check["category"] == "setting":
            scene_markers = ("room", "street", "market", "kitchen", "airport", "library", "track", "tree", "car", "bus", "train")
            if any(marker in lowered for marker in scene_markers):
                return "no", "fail"
            return "unknown", "unknown"
        if check["category"] == "props":
            if any(marker in lowered for marker in ("holding", "with", "carrying", "book", "bag", "ticket", "map", "window", "branch", "dish", "plate")):
                return "no", "fail"
            return "unknown", "unknown"
        if check["category"] == "wardrobe":
            if any(marker in lowered for marker in ("shirt", "coat", "jacket", "dress", "pants", "jeans", "boots", "sweater", "hat", "red", "blue", "gray", "black", "white", "brown")):
                return "no", "fail"
            return "unknown", "unknown"
        return "unknown", "unknown"

    def _run_single_image_prompt(self, image: Image.Image, prompt: str, max_new_tokens: int = 96) -> str:
        if self._vlm_type == "blip2":
            if prompt:
                inputs = self._processor(images=image, text=prompt, return_tensors="pt")
            else:
                inputs = self._processor(images=image, return_tensors="pt")
            target_device = torch.device(self.device)
            inputs = {
                key: value.to(target_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            input_length = int(inputs["input_ids"].shape[1]) if prompt and "input_ids" in inputs else 0
            new_tokens = generated[:, input_length:] if input_length and generated.shape[1] > input_length else generated
            return self._processor.decode(new_tokens[0], skip_special_tokens=True)

        inputs = self._processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        new_tokens = generated[:, input_length:] if input_length and generated.shape[1] > input_length else generated
        return self._processor.decode(new_tokens[0], skip_special_tokens=True)

    def _extract_labeled_answer(self, output: str, key: str) -> str:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
        match = pattern.search(output or "")
        if match:
            return match.group(1).strip()
        return ""

    def _normalize_binary_answer(self, answer: str) -> str:
        lowered = str(answer or "").strip().lower()
        if not lowered or lowered.startswith("answer:"):
            return "unknown"
        if re.match(r"^(yes|no|unknown)\b", lowered):
            return re.match(r"^(yes|no|unknown)\b", lowered).group(1)
        head = " ".join(lowered.split()[:3])
        if "yes" in head:
            return "yes"
        if re.search(r"\bno\b", head):
            return "no"
        if "unknown" in head:
            return "unknown"
        return "unknown"

    def _normalize_count_answer(self, answer: str) -> int | None:
        lowered = str(answer or "").strip().lower()
        if not lowered or lowered.startswith("answer:"):
            return None
        match = re.search(r"\b([0-4])\b", " ".join(lowered.split()[:8]))
        if match:
            return int(match.group(1))
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4}
        for token in lowered.split()[:8]:
            if token in word_map:
                return word_map[token]
        singular_subjects = {
            "man", "woman", "person", "student", "chef", "traveler", "runner",
            "bird", "dog", "cat", "robot", "child", "boy", "girl", "lady",
        }
        plural_markers = {"people", "men", "women", "children", "group", "crowd", "several", "many"}
        head_tokens = [token.strip(".,") for token in lowered.split()[:8]]
        if any(token in plural_markers for token in head_tokens):
            return None
        if any(token in singular_subjects for token in head_tokens):
            return 1
        if re.search(r"\b(a|one)\s+(man|woman|person|student|chef|traveler|runner|bird|dog|cat|robot|child|boy|girl|lady)\b", lowered):
            return 1
        return None

    def _parse_structured_panel_answers(
        self,
        output: str,
        checks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        parsed_checks = []
        for check in checks:
            raw_answer = self._extract_labeled_answer(output, check["key"])
            if check["category"] == "count":
                observed_count = self._normalize_count_answer(raw_answer)
                expected_count = None
                for term in check["expected_terms"]:
                    if str(term).isdigit():
                        expected_count = int(term)
                        break
                if observed_count is None or expected_count is None:
                    verdict = "unknown"
                else:
                    verdict = "pass" if observed_count == expected_count else "fail"
                normalized_answer = str(observed_count) if observed_count is not None else "unknown"
            else:
                normalized_answer = self._normalize_binary_answer(raw_answer)
                if normalized_answer == "unknown":
                    verdict = "unknown"
                else:
                    verdict = "pass" if normalized_answer == "yes" else "fail"
            parsed_checks.append({
                **check,
                "raw_answer": raw_answer,
                "answer": normalized_answer,
                "verdict": verdict,
                "issue_tag": STRUCTURED_TAG_BY_CATEGORY.get(check["category"]) if verdict == "fail" else "",
            })
        return parsed_checks

    def _derive_cross_panel_judgments(
        self,
        story_state: dict[str, Any],
        panel_judgments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        panel_map = {int(item.get("panel_id", 0) or 0): item for item in panel_judgments}
        judgments = []
        for link in story_state.get("cross_panel_links", []) or []:
            if not isinstance(link, dict):
                continue
            from_panel = int(link.get("from_panel", 0) or 0)
            to_panel = int(link.get("to_panel", 0) or 0)
            from_judgment = panel_map.get(from_panel)
            to_judgment = panel_map.get(to_panel)
            if not from_judgment or not to_judgment:
                continue

            issue_tags = set()
            checks = []
            from_checks = {item["key"]: item for item in from_judgment.get("checks", [])}
            to_checks = {item["key"]: item for item in to_judgment.get("checks", [])}

            if link.get("same_scene_segment"):
                from_setting = next((item for item in from_checks.values() if item["category"] == "setting"), None)
                to_setting = next((item for item in to_checks.values() if item["category"] == "setting"), None)
                if (
                    from_setting
                    and to_setting
                    and from_setting.get("verdict") == "pass"
                    and to_setting.get("verdict") == "fail"
                ):
                    issue_tags.add("vj_scene_overwrite")
                    checks.append({
                        "category": "setting",
                        "details": f"panel {to_panel} failed same-scene setting check after panel {from_panel} passed it",
                    })

            if link.get("carry_over_props"):
                from_props = next((item for item in from_checks.values() if item["category"] == "props"), None)
                to_props = next((item for item in to_checks.values() if item["category"] == "props"), None)
                if (
                    from_props
                    and to_props
                    and from_props.get("verdict") == "pass"
                    and to_props.get("verdict") == "fail"
                ):
                    issue_tags.add("vj_relational_prop_loss")
                    checks.append({
                        "category": "props",
                        "details": f"panel {to_panel} lost a carry-over prop from panel {from_panel}",
                    })

            identity_targets = {str(name).strip().lower() for name in (link.get("identity_lock_targets", []) or []) if str(name).strip()}
            for target in identity_targets:
                from_wardrobe = next(
                    (
                        item for item in from_checks.values()
                        if item["category"] == "wardrobe" and str(item.get("subject", "")).strip().lower() == target
                    ),
                    None,
                )
                to_wardrobe = next(
                    (
                        item for item in to_checks.values()
                        if item["category"] == "wardrobe" and str(item.get("subject", "")).strip().lower() == target
                    ),
                    None,
                )
                if (
                    from_wardrobe
                    and to_wardrobe
                    and from_wardrobe.get("verdict") == "pass"
                    and to_wardrobe.get("verdict") == "fail"
                ):
                    issue_tags.add("vj_wardrobe_drift")
                    checks.append({
                        "category": "wardrobe",
                        "details": f"{target} failed wardrobe continuity on panel {to_panel}",
                    })

            if checks:
                judgments.append({
                    "from_panel": from_panel,
                    "to_panel": to_panel,
                    "issue_tags": sorted(issue_tags),
                    "checks": checks,
                })
        return judgments

    def _run_structured_panel_judge(
        self,
        images: List[Image.Image],
        story_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.is_available() or not story_state:
            return {
                "model": self._vlm_type or "none",
                "confidence": "low",
                "unknown_rate": 1.0,
                "issue_tags": [],
                "panel_judgments": [],
                "cross_panel_judgments": [],
            }

        selected_questions = self._select_structured_panel_questions(story_state)
        panel_judgments = []
        total_checks = 0
        unknown_checks = 0
        issue_tags = set()

        for panel_index, image in enumerate(images, start=1):
            checks = selected_questions.get(panel_index, [])
            if not checks:
                continue
            caption = self._run_single_image_prompt(
                image,
                "",
                max_new_tokens=64,
            )
            parsed_checks = []
            for check in checks:
                normalized_answer, verdict = self._infer_check_from_caption(caption, check)
                parsed_checks.append({
                    **check,
                    "raw_answer": caption,
                    "answer": normalized_answer,
                    "verdict": verdict,
                    "issue_tag": STRUCTURED_TAG_BY_CATEGORY.get(check["category"]) if verdict == "fail" else "",
                })
            total_checks += len(parsed_checks)
            unknown_checks += sum(1 for item in parsed_checks if item["verdict"] == "unknown")
            panel_issue_tags = sorted({
                item["issue_tag"]
                for item in parsed_checks
                if item.get("issue_tag")
            })
            issue_tags.update(panel_issue_tags)
            panel_judgments.append({
                "panel_id": panel_index,
                "matched_count": sum(1 for item in parsed_checks if item["verdict"] == "pass"),
                "failed_count": sum(1 for item in parsed_checks if item["verdict"] == "fail"),
                "unknown_count": sum(1 for item in parsed_checks if item["verdict"] == "unknown"),
                "issue_tags": panel_issue_tags,
                "checks": parsed_checks,
            })

        unknown_rate = (unknown_checks / total_checks) if total_checks else 1.0
        confidence = "low" if total_checks == 0 or unknown_rate > 0.5 else "medium" if unknown_rate > 0.25 else "high"
        cross_panel_judgments = self._derive_cross_panel_judgments(story_state, panel_judgments)
        if confidence != "low":
            for item in cross_panel_judgments:
                issue_tags.update(item.get("issue_tags", []))

        return {
            "model": self._vlm_type or "none",
            "confidence": confidence,
            "unknown_rate": round(unknown_rate, 4),
            "issue_tags": sorted(issue_tags) if confidence != "low" else [],
            "panel_judgments": panel_judgments,
            "cross_panel_judgments": cross_panel_judgments,
        }

    def _apply_structured_judge_findings(self, report: VisionDiagnosticReport) -> None:
        vision_judge = report.vision_judge or {}
        if vision_judge.get("confidence") == "low":
            return

        dimension_caps = {
            "character_consistency": 5.0,
            "scene_coherence": 5.0,
            "prompt_alignment": 5.0,
        }
        for panel in vision_judge.get("panel_judgments", []):
            panel_id = panel.get("panel_id")
            for check in panel.get("checks", []):
                if check.get("verdict") != "fail":
                    continue
                category = check.get("category")
                expected = ", ".join(check.get("expected_terms", [])[:2])
                if category == "count":
                    report.add_finding(
                        "prompt_alignment",
                        f"Panel {panel_id} failed image-grounded count check for expected count {expected}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["character_consistency"] = min(dimension_caps["character_consistency"], 3.0)
                elif category == "setting":
                    report.add_finding(
                        "scene_coherence",
                        f"Panel {panel_id} failed image-grounded setting check for {expected}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["scene_coherence"] = min(dimension_caps["scene_coherence"], 3.0)
                elif category == "props":
                    report.add_finding(
                        "prompt_alignment",
                        f"Panel {panel_id} failed image-grounded prop check for {expected}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["prompt_alignment"] = min(dimension_caps["prompt_alignment"], 3.0)
                elif category == "wardrobe":
                    subject = check.get("subject", "character")
                    report.add_finding(
                        "character_consistency",
                        f"Panel {panel_id} failed image-grounded wardrobe check for {subject}: {expected}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["character_consistency"] = min(dimension_caps["character_consistency"], 3.0)

        for continuity in vision_judge.get("cross_panel_judgments", []):
            from_panel = continuity.get("from_panel")
            to_panel = continuity.get("to_panel")
            for tag in continuity.get("issue_tags", []):
                if tag == "vj_scene_overwrite":
                    report.add_finding(
                        "scene_coherence",
                        f"Panel {to_panel} failed same-scene continuity after panel {from_panel}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["scene_coherence"] = min(dimension_caps["scene_coherence"], 2.5)
                elif tag == "vj_relational_prop_loss":
                    report.add_finding(
                        "prompt_alignment",
                        f"Panel {to_panel} dropped a carry-over prop after panel {from_panel}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["prompt_alignment"] = min(dimension_caps["prompt_alignment"], 2.5)
                elif tag == "vj_wardrobe_drift":
                    report.add_finding(
                        "character_consistency",
                        f"Panel {to_panel} failed wardrobe continuity after panel {from_panel}.",
                        is_issue=True,
                        severity="info",
                    )
                    dimension_caps["character_consistency"] = min(dimension_caps["character_consistency"], 2.5)

        for dimension, cap in dimension_caps.items():
            current = report.dimensions[dimension]["score"]
            if current == 0.0:
                report.set_dimension_score(dimension, cap)
            else:
                report.set_dimension_score(dimension, min(current, cap))
    
    def evaluate_story(
        self,
        images: List[Image.Image],
        prompts: List[str],
        story_id: str,
        version_tag: str,
        config_snapshot: dict = None,
        clip_score: float = None,
        consistency_score: float = None,
        story_state: dict = None,
    ) -> VisionDiagnosticReport:
        """
        Run multi-dimensional diagnostic evaluation on a story.
        
        Args:
            images: List of 3 PIL Images (story frames)
            prompts: List of 3 text prompts (one per frame)
            story_id: Story identifier
            version_tag: Version identifier (e.g., "sca_v1", "face_v2")
            config_snapshot: Current generation config for traceability
            clip_score: CLIP score if available
            consistency_score: LPIPS consistency if available
            
        Returns:
            VisionDiagnosticReport with multi-dimensional findings
        """
        report = VisionDiagnosticReport(
            story_id=story_id,
            version_tag=version_tag,
            config_snapshot=config_snapshot or {},
        )
        
        if clip_score is not None:
            report.set_quantitative(clip_score, consistency_score or 0.0)

        if story_state:
            report.state_alignment = evaluate_prompt_state_alignment(story_state, prompts)
            if report.state_alignment.get("issues"):
                for issue in report.state_alignment["issues"][:5]:
                    report.add_finding("prompt_alignment", issue, is_issue=True, severity="info")
            report.vision_judge = self._run_structured_panel_judge(images, story_state)

        if self.is_available():
            self._evaluate_with_vlm(images, prompts, report, story_state=story_state)
        else:
            self._evaluate_fallback(images, prompts, report)

        self._apply_structured_judge_findings(report)

        if report.state_alignment:
            state_prompt_score = max(1.0, min(5.0, 5.0 * report.state_alignment.get("score", 0.0)))
            current_prompt_score = report.dimensions["prompt_alignment"]["score"]
            if current_prompt_score == 0.0:
                report.set_dimension_score("prompt_alignment", state_prompt_score)
            else:
                report.set_dimension_score("prompt_alignment", min(current_prompt_score, state_prompt_score))
        
        # Generate summary
        avg_score = sum(v["score"] for v in report.dimensions.values()) / len(report.dimensions)
        report.summary = (
            f"Story {story_id} ({version_tag}): "
            f"Avg diagnostic score {avg_score:.1f}/5.0, "
            f"{len(report.critical_issues)} critical issues"
        )
        report.failure_tags = derive_failure_tags(
            report.state_alignment,
            vision_summary=f"{report.summary} {' '.join(report.critical_issues)}",
        )
        
        # Save to version tracker
        self.version_tracker.save_report(report)
        
        return report
    
    def _evaluate_with_vlm(
        self,
        images: List[Image.Image],
        prompts: List[str],
        report: VisionDiagnosticReport,
        story_state: dict = None,
    ):
        """Use Qwen2-VL for multi-dimensional diagnostic."""
        state_contract = ""
        if story_state:
            questions = build_state_eval_questions(story_state)[:12]
            if questions:
                lines = []
                for item in questions:
                    terms = ", ".join(item.get("expected_terms", [])[:3])
                    lines.append(f"- Panel {item.get('panel_id', '?')} [{item.get('category', 'state')}]: {item.get('question', '')} Expected: {terms}")
                state_contract = (
                    "\nStoryState contract:\n"
                    + "\n".join(lines)
                    + "\nPrioritize whether the images satisfy this state contract, not only whether they loosely match the prompts."
                )

        messages = [
            {
                "role": "system",
                "content": """You are a storyboard quality inspector. Analyze the 3 frames of a visual story.
For each frame, check:
1. Character Consistency: Do characters' appearance (hair, clothes, face) stay the same across frames?
2. Scene Coherence: Does the background/setting remain consistent across frames?
3. Prompt Alignment: Does each frame match its caption accurately?
4. Temporal Flow: Does the action sequence make logical sense across frames?
5. Style Uniformity: Is the artistic style/photography consistent?
6. Artifact Quality: Any visible artifacts, blur, distortion?

For each dimension, provide:
- Score (1-5, where 5=perfect)
- Observations: one concise sentence about what is visible
- Issues: one concise sentence naming any drift, mismatch, artifact, or "none" if clean
Never copy template placeholders or bracketed text."""
            }
        ]
        
        # Add images with prompts
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            import base64
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
                    {"type": "text", "text": f"Frame {i+1}: {prompt}"}
                ]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"""Please evaluate all 3 frames together.{state_contract}

Output in this EXACT format:
## Character Consistency: 4/5
Observations: consistent clothing and face shape across frames.
Issues: slight facial drift in frame 3.

## Scene Coherence: 4/5
Observations: same classroom background is preserved.
Issues: window position shifts slightly.

## Prompt Alignment: 4/5
Observations: actions match the captions.
Issues: frame 2 under-shows the book.

## Temporal Flow: 4/5
Observations: the three panels read as a coherent sequence.
Issues: transition into the last panel feels abrupt.

## Style Uniformity: 4/5
Observations: rendering style stays photorealistic.
Issues: none.

## Artifact Quality: 4/5
Observations: images are sharp and readable.
Issues: mild blur in the background.

## Summary
One sentence overall verdict."""}
            ]
        })
        
        try:
            if self._vlm_type == "blip2":
                output = self._run_blip2_story_eval(images, prompts, state_contract)
            else:
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(text=[text], images=images, padding=True, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                gen = self._model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
                output = self._processor.decode(gen[0], skip_special_tokens=True)

            if self._is_placeholder_output(output):
                print("[VisionEval] Placeholder-style VLM output detected, falling back to deterministic analysis")
                self._evaluate_fallback(images, prompts, report)
                return

            self._parse_vlm_output(output, report)
            
        except Exception as e:
            print(f"[VisionEval] VLM inference error: {e}")
            self._evaluate_fallback(images, prompts, report)

    def _build_storyboard_image(self, images: List[Image.Image], target_height: int = 384) -> Image.Image:
        """Create a compact horizontal storyboard for single-image VLM backends."""
        resized = []
        for image in images:
            width, height = image.size
            if height == 0:
                continue
            scaled_width = max(1, int(width * (target_height / height)))
            resized.append(image.resize((scaled_width, target_height), Image.LANCZOS))

        if not resized:
            raise ValueError("No images available for storyboard")

        spacing = 8
        total_width = sum(img.width for img in resized) + spacing * (len(resized) - 1)
        canvas = Image.new("RGB", (total_width, target_height), color=(32, 32, 32))

        offset = 0
        for img in resized:
            canvas.paste(img, (offset, 0))
            offset += img.width + spacing
        return canvas

    def _run_blip2_story_eval(
        self,
        images: List[Image.Image],
        prompts: List[str],
        state_contract: str,
    ) -> str:
        """Evaluate the whole story with BLIP-2 using a storyboard image + text prompt."""
        storyboard = self._build_storyboard_image(images)
        frame_prompt_text = "\n".join(
            f"Frame {index + 1}: {prompt}" for index, prompt in enumerate(prompts)
        )
        text_prompt = (
            "You are a storyboard quality inspector. Analyze the attached storyboard of 3 story frames. "
            "Score each dimension from 1 to 5 and mention concrete issues like face drift, count drift, "
            "scene overwrite, layout failure, relational prop loss, or visible artifacts.\n\n"
            f"{frame_prompt_text}{state_contract}\n\n"
            "Reply in this EXACT format:\n"
            "## Character Consistency: 4/5\n"
            "Observations: consistent clothing and face shape across frames.\n"
            "Issues: slight facial drift in frame 3.\n\n"
            "## Scene Coherence: 4/5\n"
            "Observations: same classroom background is preserved.\n"
            "Issues: window position shifts slightly.\n\n"
            "## Prompt Alignment: 4/5\n"
            "Observations: actions match the captions.\n"
            "Issues: frame 2 under-shows the book.\n\n"
            "## Temporal Flow: 4/5\n"
            "Observations: the three panels read as a coherent sequence.\n"
            "Issues: transition into the last panel feels abrupt.\n\n"
            "## Style Uniformity: 4/5\n"
            "Observations: rendering style stays photorealistic.\n"
            "Issues: none.\n\n"
            "## Artifact Quality: 4/5\n"
            "Observations: images are sharp and readable.\n"
            "Issues: mild blur in the background.\n\n"
            "## Summary\n"
            "One sentence overall verdict."
        )

        inputs = self._processor(images=storyboard, text=text_prompt, return_tensors="pt")
        target_device = torch.device(self.device)
        inputs = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generated = self._model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        input_length = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        new_tokens = generated[:, input_length:] if input_length and generated.shape[1] > input_length else generated
        return self._processor.decode(new_tokens[0], skip_special_tokens=True)

    def _is_placeholder_output(self, output: str) -> bool:
        """Detect low-trust template-echo outputs from smaller VLMs."""
        lowered = (output or "").lower()
        return any(
            placeholder in lowered
            for placeholder in (
                "[findings and issues]",
                "[key takeaway]",
                "findings and issues",
            )
        )
    
    def _parse_vlm_output(self, output: str, report: VisionDiagnosticReport):
        """Parse structured VLM output into report dimensions."""
        dim_map = {
            "character_consistency": ["Character Consistency", r"Character Consistency:\s*(\d+(?:\.\d+)?)/5"],
            "scene_coherence": ["Scene Coherence", r"Scene Coherence:\s*(\d+(?:\.\d+)?)/5"],
            "prompt_alignment": ["Prompt Alignment", r"Prompt Alignment:\s*(\d+(?:\.\d+)?)/5"],
            "temporal_flow": ["Temporal Flow", r"Temporal Flow:\s*(\d+(?:\.\d+)?)/5"],
            "style_uniformity": ["Style Uniformity", r"Style Uniformity:\s*(\d+(?:\.\d+)?)/5"],
            "artifact_quality": ["Artifact Quality", r"Artifact Quality:\s*(\d+(?:\.\d+)?)/5"],
        }
        
        current_dim = None
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a dimension section
            for dim_key, (dim_name, score_pat) in dim_map.items():
                score_match = re.search(score_pat, line)
                if score_match:
                    score = float(score_match.group(1))
                    report.set_dimension_score(dim_key, score)
                    current_dim = dim_key
                    break
                if dim_name in line and ":" in line:
                    current_dim = dim_key
                    break
            else:
                # This is a content line, add to current dimension
                if current_dim and len(line) > 10:
                    payload = re.sub(r"^(observations?|issues?):\s*", "", line, flags=re.IGNORECASE).strip()
                    if not payload:
                        continue
                    is_issue = line.lower().startswith("issues:") or any(kw in payload.lower() for kw in
                        ["issue", "problem", "inconsistent", "artifact", "blur",
                         "different", "changed", "missing", "error", "drift",
                         "mismatch", "not match", "should be"])
                    severity = "critical" if any(kw in line.lower() for kw in 
                        ["critical", "severe", "completely"]) else \
                               "high" if any(kw in line.lower() for kw in 
                        ["significant", "major", "very"]) else "info"
                    report.add_finding(current_dim, payload, is_issue=is_issue, severity=severity)
        
        # Summary extraction
        summary_match = re.search(r"## Summary\s*\n(.+)", output, re.DOTALL)
        if summary_match:
            report.summary = summary_match.group(1).strip()[:500]
    
    def _evaluate_fallback(
        self, images: List[Image.Image], prompts: List[str], report: VisionDiagnosticReport
    ):
        """Fallback analysis using pixel-level metrics when VLM unavailable."""
        import numpy as np
        
        frames = [np.array(img.resize((224, 224))) for img in images]
        
        # Pixel-level consistency check
        if len(frames) >= 2:
            diffs = []
            for i in range(len(frames) - 1):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i + 1].astype(float)))
                diffs.append(diff)
            avg_diff = np.mean(diffs)
            
            if avg_diff < 5:
                report.add_finding("character_consistency", 
                    f"Very low frame-to-frame pixel difference ({avg_diff:.1f}) — images may be nearly identical",
                    is_issue=True, severity="high")
                report.set_dimension_score("character_consistency", 2.0)
            elif avg_diff < 20:
                report.add_finding("character_consistency",
                    f"Moderate frame-to-frame pixel difference ({avg_diff:.1f}) — reasonable variation",
                    is_issue=False)
                report.set_dimension_score("character_consistency", 3.5)
            else:
                report.add_finding("character_consistency",
                    f"High frame-to-frame pixel difference ({avg_diff:.1f}) — possible inconsistency",
                    is_issue=True, severity="info")
                report.set_dimension_score("character_consistency", 2.5)
            
            # Scene coherence: check if all frames have similar color distributions
            color_means = [f.mean(axis=(0, 1)) for f in frames]
            color_stds = [f.std(axis=(0, 1)) for f in frames]
            mean_var = np.mean([np.std([cm[c] for cm in color_means]) for c in range(3)])
            if mean_var > 30:
                report.add_finding("scene_coherence",
                    f"High color variation across frames (σ={mean_var:.1f}) — scenes may differ too much",
                    is_issue=True, severity="info")
                report.set_dimension_score("scene_coherence", 3.0)
            else:
                report.set_dimension_score("scene_coherence", 4.0)
        
        # Prompt alignment: basic check via frame brightness (proxy)
        for i, frame in enumerate(frames):
            brightness = frame.mean()
            if brightness < 30:
                report.add_finding("artifact_quality",
                    f"Frame {i+1} is very dark (brightness={brightness:.0f}) — possible generation issue",
                    is_issue=True, severity="high")
            report.set_dimension_score("prompt_alignment", 3.0)
            report.set_dimension_score("temporal_flow", 3.0)
            report.set_dimension_score("style_uniformity", 3.0)
            report.set_dimension_score("artifact_quality", 3.5)


def _compact_vision_judge(report: VisionDiagnosticReport) -> dict[str, Any]:
    vision_judge = report.vision_judge or {}
    return {
        "model": vision_judge.get("model", "none"),
        "confidence": vision_judge.get("confidence", "low"),
        "unknown_rate": float(vision_judge.get("unknown_rate", 1.0) or 1.0),
        "issue_tags": list(vision_judge.get("issue_tags", []) or []),
        "panel_judgments": [
            {
                "panel_id": item.get("panel_id"),
                "matched_count": item.get("matched_count", 0),
                "failed_count": item.get("failed_count", 0),
                "unknown_count": item.get("unknown_count", 0),
                "issue_tags": item.get("issue_tags", []),
            }
            for item in (vision_judge.get("panel_judgments", []) or [])
        ],
        "cross_panel_judgments": [
            {
                "from_panel": item.get("from_panel"),
                "to_panel": item.get("to_panel"),
                "issue_tags": item.get("issue_tags", []),
            }
            for item in (vision_judge.get("cross_panel_judgments", []) or [])
        ],
    }


def _write_story_vision_judge(story_dir: Path, report: VisionDiagnosticReport) -> None:
    eval_file = story_dir / "evaluation.json"
    if not eval_file.exists():
        return
    with open(eval_file, encoding="utf-8") as handle:
        eval_data = json.load(handle)
    eval_data["vision_judge"] = _compact_vision_judge(report)
    with open(eval_file, "w", encoding="utf-8") as handle:
        json.dump(eval_data, handle, indent=2)


def evaluate_all_stories(
    base_dir: str = None,
    version_tag: str = "sca_v1",
    story_ids: List[str] = None,
    config: dict = None,
):
    """
    Batch evaluate all generated stories with VLM.
    
    Args:
        base_dir: Directory with story folders (each containing frame_*.png)
        version_tag: Version identifier
        story_ids: List of story subdirectories to evaluate (or all)
        config: Generation config snapshot
    
    Returns:
        Markdown summary
    """
    evaluator = VisionEvaluator()
    tracker = VersionTracker()
    if base_dir is None:
        base_dir = str(STORYGEN_DIR / "outputs/taskA_batch")
    base_path = Path(base_dir)
    
    if story_ids is None:
        story_ids = sorted([
            d.name for d in base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "versions"
        ])
    
    results = []
    for sid in story_ids:
        story_dir = base_path / sid
        images = []
        prompts = []
        story_state = {}
        
        # Load frames in order
        for fname in sorted(story_dir.glob("frame_*.png")):
            images.append(Image.open(fname))
        
        # Try to load prompts from evaluation.json / render_plan.json
        eval_file = story_dir / "evaluation.json"
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            prompts = eval_data.get("prompts", [f"Frame {i+1}" for i in range(len(images))])
        else:
            prompts = [f"Frame {i+1}" for i in range(len(images))]

        render_plan_file = story_dir / "render_plan.json"
        if render_plan_file.exists():
            with open(render_plan_file) as f:
                render_plan = json.load(f)
            if render_plan:
                prompts = [
                    item.get("compiled_prompt", prompts[i] if i < len(prompts) else f"Frame {i+1}")
                    for i, item in enumerate(render_plan)
                ]

        story_state_file = story_dir / "story_state.json"
        if story_state_file.exists():
            with open(story_state_file) as f:
                story_state = json.load(f)
        else:
            board_file = story_dir / "production_board.json"
            if board_file.exists():
                with open(board_file) as f:
                    board_data = json.load(f)
                story_state = board_data.get("story_state", {})
        
        if not images:
            continue
        
        # Run evaluation
        clip_score = None
        consistency_score = None
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            clip_score = eval_data.get("avg_clip_score")
            consistency_score = eval_data.get("avg_consistency")
        
        report = evaluator.evaluate_story(
            images=images, prompts=prompts,
            story_id=sid, version_tag=version_tag,
            config_snapshot=config or {},
            clip_score=clip_score, consistency_score=consistency_score,
            story_state=story_state,
        )
        _write_story_vision_judge(story_dir, report)
        results.append(report)
    
    try:
        from storygen.evaluation_hub.failure_atlas import build_failure_atlas
        atlas = build_failure_atlas(base_path.iterdir())
        with open(base_path / "failure_atlas.json", "w", encoding="utf-8") as handle:
            json.dump(atlas, handle, indent=2)
    except Exception as exc:
        print(f"[VisionEval] Failed to refresh failure_atlas.json: {exc}")

    # Generate summary table
    output = tracker.summary_table()
    
    return output


if __name__ == "__main__":
    import sys
    story_id = sys.argv[1] if len(sys.argv) > 1 else "01"
    base = sys.argv[2] if len(sys.argv) > 2 else str(STORYGEN_DIR / "outputs/taskA_batch")
    version = sys.argv[3] if len(sys.argv) > 3 else "quick_test"
    
    result = evaluate_all_stories(
        base_dir=base,
        version_tag=version,
        story_ids=[story_id],
    )
    print(result)
