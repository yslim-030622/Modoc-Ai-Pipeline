from pathlib import Path
import tempfile
import unittest

from PIL import Image

from modoc_pipeline.excel_source import load_qna_sources
from modoc_pipeline.grounding import generate_grounding_report
from modoc_pipeline.quality import deterministic_quality_report
from modoc_pipeline.renderer import bake_meme_text_onto_images
from modoc_pipeline.schemas import GroundingReport, MemePlan, ScriptPackage


class StaticValidationTests(unittest.TestCase):
    def test_grounding_skip_returns_expert_only_report(self):
        source = load_qna_sources(Path("Q&A Blog Contents List.xlsx"), row_number=7)[0]
        report = generate_grounding_report(
            api_key="unused",
            model="gemini-3.5-flash",
            source=source,
            enable_search=False,
        )
        parsed = GroundingReport.model_validate(report.parsed)
        self.assertEqual(parsed.status, "expert_only")
        self.assertTrue(parsed.supported_facts)

    def test_script_package_requires_claim_evidence(self):
        payload = {
            "scripts": {
                key: {
                    "language": key.title(),
                    "title": "Title",
                    "hook": "Hook",
                    "body": ["Body"],
                    "safety_caveat": "Ask a clinician if symptoms worsen.",
                    "cta": "Follow for more.",
                    "estimated_seconds": 20,
                }
                for key in ("english", "korean", "spanish")
            },
            "medical_claims": [
                {
                    "claim": "Unsupported claim",
                    "evidence_from_expert_answer": "",
                    "grounding_fact_indices": [],
                    "citation_indices": [],
                    "appears_in_languages": ["english"],
                    "risk_level": "medium",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "lacks evidence"):
            ScriptPackage.model_validate(payload)

    def test_deterministic_quality_gate_passes_valid_script(self):
        report = deterministic_quality_report(
            "script_package",
            {
                "medical_claims": [
                    {
                        "claim": "Supported claim",
                        "evidence_from_expert_answer": "The expert answer states this.",
                        "grounding_fact_indices": [],
                        "citation_indices": [],
                        "appears_in_languages": ["english"],
                        "risk_level": "low",
                    }
                ],
            },
        )
        self.assertEqual(report.status, "passed")

    def test_deterministic_quality_gate_rejects_unsupported_claim(self):
        report = deterministic_quality_report(
            "script_package",
            {
                "medical_claims": [
                    {
                        "claim": "Unsupported claim",
                        "evidence_from_expert_answer": "",
                        "grounding_fact_indices": [],
                        "citation_indices": [],
                        "appears_in_languages": ["english"],
                        "risk_level": "medium",
                    }
                ],
            },
        )
        self.assertEqual(report.status, "failed")
        self.assertTrue(any("evidence" in issue.issue.lower() for issue in report.issues))

    def test_meme_plan_schema_accepts_legacy_payload_without_creative_fields(self):
        lang_plan = {
            "language": "english",
            "cultural_context": "Parenting short-form",
            "trending_hooks": ["POV"],
            "visual_style_anchor": "modern flat illustration",
            "character_sheet": "tired parent, teal sweatshirt, black leggings",
            "scenes": [
                {
                    "scene_id": "scene_01",
                    "meme_format": "pov",
                    "top_text": "POV:",
                    "bottom_text": "The key detail matters",
                    "image_prompt": "modern flat illustration. tired parent, teal sweatshirt, black leggings. Looking at a phone.",
                    "duration_seconds": 3.0,
                    "tts_text": "The key detail matters.",
                },
                {
                    "scene_id": "scene_02",
                    "meme_format": "pov",
                    "bottom_text": "Not every symptom means panic",
                    "image_prompt": "modern flat illustration. tired parent, teal sweatshirt, black leggings. Taking a breath.",
                },
                {
                    "scene_id": "scene_03",
                    "meme_format": "pov",
                    "bottom_text": "Ask when symptoms change",
                    "image_prompt": "modern flat illustration. tired parent, teal sweatshirt, black leggings. Calmly checking notes.",
                },
            ],
        }
        payload = {
            "english": lang_plan,
            "korean": {**lang_plan, "language": "korean"},
            "spanish": {**lang_plan, "language": "spanish"},
            "source_topic": "pediatric health",
        }

        parsed = MemePlan.model_validate(payload)
        self.assertEqual(parsed.english.caption_style, "impact")
        self.assertEqual(parsed.english.scenes[0].caption_role, "hook")

    def test_caption_styles_render_baked_images(self):
        scenes = [
            {
                "scene_id": "scene_01",
                "top_text": "Nobody warns you:",
                "bottom_text": "This caption is intentionally long but should fit",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            baked = root / "baked"
            raw.mkdir()
            Image.new("RGB", (1080, 1920), (40, 80, 100)).save(raw / "scene_01.png")

            for style in ("impact", "clean_reels", "korean_jjal", "spanish_social"):
                paths = bake_meme_text_onto_images(
                    scenes=scenes,
                    raw_images_dir=raw,
                    baked_dir=baked / style,
                    language="english",
                    caption_style=style,
                )
                self.assertEqual(len(paths), 1)
                self.assertTrue(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
