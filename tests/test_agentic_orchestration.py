import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modoc_pipeline.cli import build_parser
from modoc_pipeline.dashboard import build_runs_payload
from modoc_pipeline.imagen_client import ImagenConfig, generate_meme_images, _review_scene_image
from modoc_pipeline.meme_planner import (
    _build_candidate_prompt,
    _build_plan_prompt,
    _build_visual_brief_prompt,
    _load_trend_library,
    _merge_visual_grammar,
    _parse_candidate_set,
    _parse_meme_plan,
    _parse_score_set,
    _repair_plan_guardrails,
    _score_trend_item,
    _select_trends_from_library,
)
import modoc_pipeline.orchestration.graph as graph_module
from modoc_pipeline.orchestration.nodes import bgm_agent_node, route_after_failure, tts_agent_node
from modoc_pipeline.orchestration.state import PipelineState, ReviewReport
from modoc_pipeline.prompt_profiles import load_campaign_profile
from modoc_pipeline.reviewers import (
    assert_meme_plan_quality,
    assert_bgm_complete,
    assert_tts_complete,
    deterministic_creative_report,
    deterministic_meme_text_report,
    deterministic_visual_prompt_report,
)


class AgenticOrchestrationTests(unittest.TestCase):
    def test_pipeline_state_minimal_payload_is_json_serializable(self):
        state: PipelineState = {
            "row_number": 7,
            "input_path": "Q&A Blog Contents List.xlsx",
            "output_dir": "logs",
            "log_path": "logs/pipeline_runs.csv",
            "video_dir": "videos",
            "languages": None,
            "agent_trace": [],
        }
        encoded = json.dumps(state)
        self.assertIn('"row_number": 7', encoded)

    def test_review_report_schema_accepts_failed_payload(self):
        report = ReviewReport.model_validate(
            {
                "status": "failed",
                "stage": "medical_review",
                "summary": "Unsupported reassurance.",
                "issues": [{"severity": "high", "issue": "Too reassuring"}],
                "repair_instruction": "Restore source uncertainty.",
            }
        )
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.repair_instruction, "Restore source uncertainty.")

    def test_meme_plan_quality_assertion_does_not_fail_closed_on_visual_only_review(self):
        assert_meme_plan_quality(
            {
                "creative_review": {"status": "passed", "stage": "creative_review"},
                "meme_plan_review": {"status": "passed", "stage": "meme_plan_review"},
                "visual_prompt_review": {
                    "status": "failed",
                    "stage": "visual_prompt_review",
                    "summary": "Deterministic visual prompt validation failed.",
                },
            }
        )

    def test_meme_plan_quality_assertion_still_fails_on_text_review(self):
        with self.assertRaises(ValueError):
            assert_meme_plan_quality(
                {
                    "creative_review": {"status": "passed", "stage": "creative_review"},
                    "meme_plan_review": {
                        "status": "failed",
                        "stage": "meme_plan_review",
                        "summary": "Deterministic meme text validation failed.",
                    },
                    "visual_prompt_review": {"status": "passed", "stage": "visual_prompt_review"},
                }
            )

    def test_visual_prompt_validation_rejects_missing_anchor_and_character(self):
        report = deterministic_visual_prompt_report(
            {
                "english": {
                    "visual_style_anchor": "modern flat illustration",
                    "character_sheet": "tired parent, teal sweatshirt, black leggings",
                    "scenes": [
                        {
                            "scene_id": "scene_01",
                            "image_prompt": "A parent looks at a phone with a big logo.",
                        }
                    ],
                }
            }
        )
        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("visual_style_anchor", joined)
        self.assertIn("character_sheet", joined)

    def test_route_after_failure_continues_when_no_failure(self):
        self.assertEqual(route_after_failure({}), "continue")

    def test_route_after_failure_fails_closed_when_status_set(self):
        self.assertEqual(route_after_failure({"failure_status": "failed"}), "fail_closed")

    def test_creative_review_rejects_missing_metadata_and_low_safety_score(self):
        report = deterministic_creative_report(
            {
                "english": {
                    "caption_style": "impact",
                    "scenes": [
                        {"caption_role": "hook"},
                        {"caption_role": "tension"},
                        {"caption_role": "insight"},
                        {"caption_role": "relief"},
                    ],
                }
            },
            {"english": [{"selected": True, "medical_safety": 3}]},
        )
        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("creative_angle", joined)
        self.assertIn("medical_safety", joined)

    def test_creative_review_allows_bright_bgm_but_rejects_missing_instrumental_guardrail(self):
        plan = {
            "english": {
                "creative_angle": "Parent notices postpartum swelling",
                "trend_rationale": "Real parent concern",
                "caption_style": "clean_reels",
                "tts_style": "calm",
                "bgm_prompt": "Bright short-form pop groove with a catchy first bar",
                "bgm_config": {"density": 0.72, "brightness": 0.82},
                "scenes": [
                    {"caption_role": "hook"},
                    {"caption_role": "tension"},
                    {"caption_role": "insight"},
                    {"caption_role": "relief"},
                ],
            },
            "korean": {
                "creative_angle": "산후 부종",
                "trend_rationale": "현실 육아",
                "caption_style": "korean_jjal",
                "tts_style": "calm",
                "bgm_prompt": "Bright instrumental pop groove, no vocals",
                "bgm_config": {"density": 0.55, "brightness": 0.62},
                "scenes": [
                    {"caption_role": "hook"},
                    {"caption_role": "tension"},
                    {"caption_role": "insight"},
                    {"caption_role": "relief"},
                ],
            },
            "spanish": {
                "creative_angle": "Hinchazón posparto",
                "trend_rationale": "realidad",
                "caption_style": "spanish_social",
                "tts_style": "calm",
                "bgm_prompt": "Bright instrumental guitar groove, no vocals",
                "bgm_config": {"density": 0.55, "brightness": 0.62},
                "scenes": [
                    {"caption_role": "hook"},
                    {"caption_role": "tension"},
                    {"caption_role": "insight"},
                    {"caption_role": "relief"},
                ],
            },
        }
        scores = {
            lang: [{"selected": True, "medical_safety": 5}]
            for lang in ("english", "korean", "spanish")
        }
        scripts = {"english": {"title": "Postpartum swelling and blood clot concern"}}

        report = deterministic_creative_report(plan, scores, scripts=scripts)

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("instrumental", joined)
        self.assertNotIn("too playful", joined)

    def test_meme_text_review_rejects_unsupported_urgency_remedies_and_stereotypes(self):
        scripts = {"english": {"body": ["Ask your doctor about an ultrasound."]}}
        plan = {
            "korean": {
                "scenes": [
                    {
                        "scene_id": "scene_01",
                        "top_text": "산후조리원 퇴소 후:",
                        "bottom_text": "호박즙 먹고 마사지하면 빠지겠지",
                        "tts_text": "응급실에 바로 가야 할까요?",
                    }
                ]
            },
            "spanish": {
                "scenes": [
                    {
                        "scene_id": "scene_01",
                        "top_text": "Mamá latina:",
                        "bottom_text": "Ponte alcohol de romero y ya",
                        "tts_text": "Ve a urgencias ahora.",
                    }
                ]
            },
        }

        report = deterministic_meme_text_report(scripts, plan)

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("remedy", joined)
        self.assertIn("emergency", joined)
        self.assertIn("Mamá latina", joined)

    def test_meme_text_review_allows_source_supported_avoidance_language(self):
        scripts = {
            "korean": {
                "body": [
                    "집에서는 수분을 조금씩 자주 먹이시고, 미온수 마사지나 알코올 마사지는 절대 하지 마세요."
                ]
            }
        }
        plan = {
            "korean": {
                "scenes": [
                    {
                        "scene_id": "kor_scene_3",
                        "top_text": "안전한 홈케어",
                        "bottom_text": "수분은 자주, 마사지는 금지",
                        "tts_text": "수분을 조금씩 주고 마사지는 하지 마세요.",
                    }
                ]
            }
        }

        report = deterministic_meme_text_report(scripts, plan)

        self.assertEqual(report.status, "passed")

    def test_meme_text_review_does_not_treat_er_inside_words_as_urgency_support(self):
        scripts = {"english": {"body": ["Offer water if fever is present."]}}
        plan = {
            "english": {
                "scenes": [
                    {
                        "scene_id": "scene_01",
                        "top_text": "Fever check",
                        "bottom_text": "Go to the ER now",
                        "tts_text": "Use the emergency room now.",
                    }
                ]
            }
        }

        report = deterministic_meme_text_report(scripts, plan)

        self.assertEqual(report.status, "failed")
        self.assertIn("emergency-room urgency", " ".join(issue.issue for issue in report.issues))

    def test_repair_plan_guardrails_removes_unsupported_text_gate_terms(self):
        plan = {
            "korean": {
                "scenes": [
                    {
                        "scene_id": "ko_scene_1",
                        "top_text": "응급실?",
                        "bottom_text": "호박즙 먹고 마사지하면 괜찮죠",
                        "tts_text": "응급실에서 호박즙과 마사지를 물어보세요.",
                        "image_prompt": "",
                    }
                ]
            },
            "spanish": {
                "scenes": [
                    {
                        "scene_id": "es_scene_1",
                        "top_text": "Mamá",
                        "bottom_text": "Alcohol de romero",
                        "tts_text": "Ve a urgencias por remedio casero.",
                        "image_prompt": "",
                    }
                ]
            },
        }

        repaired = _repair_plan_guardrails(plan, scripts={"english": {"body": ["Call your doctor."]}})
        report = deterministic_meme_text_report({"english": {"body": ["Call your doctor."]}}, repaired)

        self.assertEqual(report.status, "passed")

    def test_generate_parser_accepts_skip_search(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "24", "--skip-search"])
        self.assertEqual(args.command, "generate")
        self.assertEqual(args.rows, [24])
        self.assertTrue(args.skip_search)

    def test_generate_parser_accepts_campaign_profile(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "24", "--campaign-profile", "profiles/current.json"])
        self.assertEqual(args.campaign_profile, "profiles/current.json")

    def test_generate_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "24"])
        self.assertEqual(args.rows, [24])
        self.assertFalse(args.skip_tts)
        self.assertFalse(args.no_zoom)

    def test_dashboard_parser_defaults_to_localhost(self):
        parser = build_parser()
        args = parser.parse_args(["dashboard"])
        self.assertEqual(args.command, "dashboard")
        self.assertEqual(args.host, "localhost")
        self.assertEqual(args.port, 8765)

    def test_meme_prompts_do_not_force_legacy_locked_templates(self):
        visual_brief_prompt = _build_visual_brief_prompt(
            scripts={"english": {"body": ["Use only the supported facts."]}},
            topic="pediatric health",
        )
        candidate_prompt = _build_candidate_prompt(
            scripts={"english": {"body": ["Use only the supported facts."]}},
            topic="pediatric health",
            trends={"english": [], "korean": [], "spanish": []},
            visual_brief={"content_type": "fever_triage", "allowed_props": ["blank digital thermometer"]},
        )
        plan_prompt = _build_plan_prompt(
            scripts={"english": {"body": ["Use only the supported facts."]}},
            topic="pediatric health",
            trends_json="{}",
            candidates={"english": [], "korean": [], "spanish": []},
            scores={"english": [], "korean": [], "spanish": []},
            visual_brief={"content_type": "fever_triage", "allowed_props": ["blank digital thermometer"]},
        )

        combined = f"{visual_brief_prompt}\n{candidate_prompt}\n{plan_prompt}"
        self.assertNotIn("Mamá latina: format", combined)
        self.assertNotIn("dark circles under eyes", combined)
        self.assertNotIn("use 새벽 육아", combined)
        self.assertIn("one video-level caregiver", combined)
        self.assertIn("ContentVisualBrief", combined)
        self.assertIn("content_type", combined)
        self.assertIn("fever_triage", combined)
        self.assertIn("VISUAL SCENE WORKFLOW", combined)
        self.assertIn("scene_visual_action", combined)
        self.assertIn("shot_type", combined)
        self.assertIn("Do not make medical infographics", combined)
        self.assertNotIn("For ALL topics", combined)
        self.assertIn("calm-bright", combined)
        self.assertIn("brightness <= 0.92", combined)
        self.assertIn("format_archetype", candidate_prompt)
        self.assertIn("why_this_row_fits", candidate_prompt)
        self.assertIn("medical_fit", candidate_prompt)

    def test_local_trend_library_loads_as_trend_research_compatible(self):
        library, path, load_error = _load_trend_library()

        self.assertTrue(path.name.endswith("trend_library.json"))
        self.assertEqual(load_error, "")

        selected = _select_trends_from_library(
            library,
            topic="Acetaminophen dosing",
            scripts={"english": {"body": ["Dose by weight, not age."]}},
            visual_brief={"content_type": "medication_dosing"},
            campaign_profile={},
        )

        self.assertEqual(set(selected), {"english", "korean", "spanish"})
        for lang, items in selected.items():
            self.assertGreaterEqual(len(items), 3, lang)
            self.assertTrue(all(item["medical_fit"] != "avoid" for item in items))
            self.assertTrue(all(item["format_archetype"] for item in items))
            self.assertTrue(all(item["why_this_row_fits"] for item in items))

    def test_trend_scorer_penalizes_unsafe_or_irrelevant_formats(self):
        safe = {
            "format_archetype": "decision_tree",
            "medical_fit": "safe",
            "caption_density": "medium",
            "example_topics": ["fever_triage"],
            "avoid_when": [],
            "confidence": 0.8,
        }
        unsafe = {
            "format_archetype": "rage_bait_hot_take",
            "medical_fit": "avoid",
            "caption_density": "high",
            "example_topics": [],
            "avoid_when": ["pediatric health"],
            "confidence": 0.8,
        }

        safe_score = _score_trend_item(safe, topic_text="pediatric health fever", content_type="fever_triage")
        unsafe_score = _score_trend_item(unsafe, topic_text="pediatric health fever", content_type="fever_triage")

        self.assertGreater(safe_score, unsafe_score)
        self.assertLess(unsafe_score, 0)

    def test_trend_library_falls_back_when_configured_file_is_invalid(self):
        with patch.dict("os.environ", {"MODOC_TREND_LIBRARY": "/tmp/modoc-missing-trends.json"}):
            library, path, load_error = _load_trend_library()

        self.assertEqual(library["version"], "fallback")
        self.assertIn("modoc-missing-trends", str(path))
        self.assertTrue(load_error)

    def test_audio_coverage_helpers_fail_when_required_outputs_are_missing(self):
        plan = {
            "english": {
                "scenes": [
                    {"scene_id": "scene_01", "tts_text": "Supported point."},
                    {"scene_id": "scene_02", "tts_text": "Safety caveat."},
                ]
            },
            "korean": {"scenes": [{"scene_id": "scene_01", "tts_text": "안전 문장."}]},
        }

        with self.assertRaisesRegex(RuntimeError, "english/scene_02"):
            assert_tts_complete(
                audio_paths={"english": {"scene_01": Path("scene_01.wav")}},
                meme_plan=plan,
                languages={"english"},
            )

        with self.assertRaisesRegex(RuntimeError, "korean"):
            assert_bgm_complete(
                bgm_paths={"english": Path("english_bgm.mp3")},
                meme_plan=plan,
                languages={"english", "korean"},
            )

    def test_tts_and_bgm_nodes_fail_closed_on_missing_outputs_but_skip_tts_allows_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "run_dir": tmp,
                "meme_plan": {
                    "english": {
                        "scenes": [{"scene_id": "scene_01", "tts_text": "Supported point."}]
                    }
                },
                "languages": ["english"],
                "api_key": "unused",
                "agent_trace": [],
            }

            with patch("modoc_pipeline.orchestration.nodes.load_tts_config", return_value=object()), patch(
                "modoc_pipeline.orchestration.nodes.generate_meme_gemini_tts",
                return_value={"english": {}},
            ):
                tts_state = tts_agent_node(base_state)

            self.assertEqual(tts_state["failure_status"], "failed")
            self.assertIn("TTS generation missing", tts_state["failure_message"])

            with patch("modoc_pipeline.orchestration.nodes.generate_all_bgm", return_value={}):
                bgm_state = bgm_agent_node(base_state)

            self.assertEqual(bgm_state["failure_status"], "failed")
            self.assertIn("BGM generation missing", bgm_state["failure_message"])

            skipped = tts_agent_node({**base_state, "skip_tts": True})
            self.assertNotIn("failure_status", skipped)
            self.assertEqual(skipped["audio_paths"], {})

    def test_image_qa_fails_closed_for_invalid_status_and_invalid_json(self):
        class FakeModels:
            def __init__(self, text: str) -> None:
                self.text = text

            def generate_content(self, **kwargs):
                return type("FakeResponse", (), {"text": self.text})()

        class FakeClient:
            def __init__(self, text: str) -> None:
                self.models = FakeModels(text)

        config = ImagenConfig(api_key="unused", model="image-model")
        scene = {"scene_id": "scene_01", "tts_text": "Supported point.", "primary_prop": "phone"}
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "scene.png"
            image_path.write_bytes(b"not-a-real-png-but-not-read-by-fake-client")

            invalid_status = _review_scene_image(
                client=FakeClient('{"status":"unclear"}'),
                config=config,
                image_path=image_path,
                scene=scene,
                visual_brief={},
            )
            invalid_json = _review_scene_image(
                client=FakeClient("not json"),
                config=config,
                image_path=image_path,
                scene=scene,
                visual_brief={},
            )

        self.assertEqual(invalid_status["status"], "failed")
        self.assertEqual(invalid_json["status"], "failed")

    def test_image_generation_continues_on_qa_failure_by_default(self):
        plan = {
            "english": {
                "visual_style_anchor": "clean illustration",
                "character_sheet": "stable caregiver",
                "scenes": [{"scene_id": "scene_01", "image_prompt": "caregiver offers water"}],
            }
        }

        def fake_generate(**kwargs):
            kwargs["output_path"].write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "modoc_pipeline.imagen_client._generate_scene_image",
            side_effect=fake_generate,
        ), patch(
            "modoc_pipeline.imagen_client._review_scene_image",
            return_value={"status": "failed", "issues": ["missing prop"], "repair_instruction": "fix prop"},
        ):
            result = generate_meme_images(
                meme_plan=plan,
                output_dir=Path(tmp),
                config=ImagenConfig(api_key="unused", model="image-model", max_regenerations=0),
                languages={"english"},
                sleep_between_requests=0,
            )

        self.assertIn("scene_01", result["english"])

    def test_image_generation_can_fail_closed_on_qa_failure(self):
        plan = {
            "english": {
                "visual_style_anchor": "clean illustration",
                "character_sheet": "stable caregiver",
                "scenes": [{"scene_id": "scene_01", "image_prompt": "caregiver offers water"}],
            }
        }

        def fake_generate(**kwargs):
            kwargs["output_path"].write_bytes(b"png")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "modoc_pipeline.imagen_client._generate_scene_image",
            side_effect=fake_generate,
        ), patch(
            "modoc_pipeline.imagen_client._review_scene_image",
            return_value={"status": "failed", "issues": ["missing prop"], "repair_instruction": "fix prop"},
        ):
            with self.assertRaises(Exception):
                generate_meme_images(
                    meme_plan=plan,
                    output_dir=Path(tmp),
                    config=ImagenConfig(
                        api_key="unused",
                        model="image-model",
                        image_qa_fail_closed=True,
                        max_regenerations=0,
                    ),
                    languages={"english"},
                    sleep_between_requests=0,
                )

    def test_visual_prompt_validation_rejects_repetitive_scene_plans(self):
        scenes = []
        for index, role in enumerate(("hook", "tension", "insight", "relief"), start=1):
            scenes.append(
                {
                    "scene_id": f"scene_{index:02d}",
                    "caption_role": role,
                    "medical_message": "Supported medical message.",
                    "scene_visual_action": "Caregiver holding the same mug.",
                    "safe_props": ["warm mug"],
                    "shot_type": "medium_action",
                    "primary_prop": "warm mug",
                    "background": "sunlit kitchen",
                    "image_prompt": (
                        "modern flat illustration. stable caregiver character. "
                        "Caregiver standing still, holding the warm mug in a sunlit kitchen."
                    ),
                }
            )
        report = deterministic_visual_prompt_report(
            {
                "english": {
                    "visual_style_anchor": "modern flat illustration",
                    "character_sheet": "stable caregiver character",
                    "scenes": scenes,
                }
            }
        )

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("primary_prop", joined)
        self.assertIn("shot_type", joined)

    def test_visual_prompt_validation_uses_content_specific_allowed_props(self):
        anchor = "modern flat illustration"
        character = "stable caregiver character"
        valid_scene = {
            "scene_id": "scene_01",
            "caption_role": "hook",
            "medical_message": "Check fever calmly.",
            "scene_visual_action": "Caregiver checks a blank thermometer near a water cup.",
            "safe_props": ["blank digital thermometer", "water cup"],
            "shot_type": "close_up",
            "primary_prop": "blank digital thermometer",
            "background": "kitchen table",
            "tts_text": "Check fever calmly.",
            "image_prompt": (
                f"{anchor}. {character}. Close-up at a kitchen table, caregiver holding "
                "a blank digital thermometer beside a water cup, calm posture, blank display."
            ),
        }
        filler_scene = {
            **valid_scene,
            "scene_id": "scene_02",
            "safe_props": ["potted plant"],
            "primary_prop": "potted plant",
            "image_prompt": (
                f"{anchor}. {character}. Medium shot watering a potted plant by a sunny window."
            ),
        }
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "fever_triage",
                    "allowed_props": ["blank digital thermometer", "water cup", "phone"],
                    "forbidden_visuals": ["thermometer numbers"],
                },
                "english": {
                    "visual_style_anchor": anchor,
                    "character_sheet": character,
                    "scenes": [valid_scene, filler_scene],
                },
            }
        )

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("generic filler", joined)
        self.assertIn("visual_brief.allowed_props", joined)

    def test_medication_visual_brief_removes_sensitive_props(self):
        brief = _merge_visual_grammar(
            {
                "content_type": "medication_dosing",
                "allowed_props": ["unlabeled medicine bottle", "oral syringe without numbers", "blank dosing note"],
                "forbidden_visuals": [],
                "must_show": ["caregiver holding an oral syringe", "blank dosing note on counter"],
                "must_not_show": [],
            }
        )

        joined_allowed = " ".join(brief["allowed_props"]).lower()
        joined_must_show = " ".join(brief["must_show"]).lower()
        joined_forbidden = " ".join(brief["forbidden_visuals"]).lower()

        self.assertNotIn("syringe", joined_allowed)
        self.assertNotIn("medicine bottle", joined_allowed)
        self.assertNotIn("syringe", joined_must_show)
        self.assertIn("syringe", joined_forbidden)
        self.assertIn("medicine bottle", joined_forbidden)

    def test_plan_guardrail_repair_replaces_sensitive_medication_visuals(self):
        plan = {
            "visual_brief": {
                "content_type": "medication_dosing",
                "allowed_props": ["blank dosing note", "phone with blank screen", "water cup"],
                "forbidden_visuals": [],
            },
            "english": {
                "bgm_prompt": "Bright pop instrumental, no vocals",
                "scenes": [
                    {
                        "scene_id": "scene_01",
                        "caption_role": "hook",
                        "top_text": "Dose",
                        "bottom_text": "Check weight",
                        "primary_prop": "oral syringe without numbers",
                        "safe_props": ["unlabeled medicine bottle", "oral syringe without numbers"],
                        "tts_text": "Check weight first.",
                        "image_prompt": "Caregiver drawing liquid medicine into an oral syringe from an unlabeled medicine bottle.",
                    }
                ],
            },
        }

        repaired = _repair_plan_guardrails(plan)
        scene = repaired["english"]["scenes"][0]
        joined = " ".join([scene["primary_prop"], " ".join(scene["safe_props"]), scene["image_prompt"]]).lower()

        self.assertIn("blank dosing note", joined)
        self.assertNotIn("oral syringe", joined)
        self.assertNotIn("medicine bottle", joined)
        self.assertNotIn("liquid medicine", joined)

    def test_visual_prompt_validation_rejects_sensitive_medication_imagery(self):
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "medication_dosing",
                    "allowed_props": ["blank dosing note", "phone with blank screen"],
                    "forbidden_visuals": ["medicine bottle", "oral syringe", "pills"],
                },
                "english": {
                    "visual_style_anchor": "clean illustration",
                    "character_sheet": "stable caregiver character",
                    "scenes": [
                        {
                            "scene_id": "scene_01",
                            "caption_role": "hook",
                            "medical_message": "Dose safely by weight.",
                            "scene_visual_action": "Caregiver prepares medication.",
                            "safe_props": ["oral syringe without numbers"],
                            "shot_type": "tabletop_action",
                            "primary_prop": "oral syringe without numbers",
                            "background": "kitchen",
                            "tts_text": "Check the weight first.",
                            "image_prompt": "clean illustration. stable caregiver character. Caregiver draws liquid into an oral syringe from a medicine bottle.",
                        }
                    ],
                },
            }
        )

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("sensitive medication imagery", joined)

    def test_visual_prompt_validation_requires_primary_prop_in_prompt(self):
        anchor = "modern flat illustration"
        character = "stable caregiver character"
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "lab_result",
                    "allowed_props": ["stethoscope", "blank folder", "doctor desk"],
                    "forbidden_visuals": [],
                },
                "english": {
                    "visual_style_anchor": anchor,
                    "character_sheet": character,
                    "scenes": [
                        {
                            "scene_id": "scene_01",
                            "caption_role": "hook",
                            "medical_message": "X-ray and stethoscope can differ.",
                            "scene_visual_action": "Caregiver compares test feedback at a doctor desk.",
                            "safe_props": ["stethoscope"],
                            "shot_type": "over_the_shoulder",
                            "primary_prop": "stethoscope",
                            "background": "doctor desk",
                            "tts_text": "The stethoscope can sound different from the X-ray.",
                            "image_prompt": f"{anchor}. {character}. Caregiver reviews a blank folder at a doctor desk.",
                        }
                    ],
                },
            }
        )

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("primary_prop", joined)
        self.assertIn("stethoscope", joined)

    def test_visual_prompt_validation_rejects_generic_emergency_warning_scene(self):
        anchor = "modern flat illustration"
        character = "stable caregiver character"
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "fever_triage",
                    "allowed_props": ["water cup", "phone", "blank digital thermometer"],
                    "forbidden_visuals": [],
                },
                "english": {
                    "visual_style_anchor": anchor,
                    "character_sheet": character,
                    "scenes": [
                        {
                            "scene_id": "scene_04",
                            "caption_role": "relief",
                            "medical_message": "Know when emergency care is needed.",
                            "scene_visual_action": "Caregiver offers water in a calm room.",
                            "safe_props": ["water cup"],
                            "shot_type": "medium_action",
                            "primary_prop": "water cup",
                            "background": "living room",
                            "tts_text": "Go to the ER immediately if unresponsive or breathing is hard.",
                            "image_prompt": f"{anchor}. {character}. Caregiver offers a water cup in a calm living room.",
                        }
                    ],
                },
            }
        )

        self.assertEqual(report.status, "failed")
        joined = " ".join(issue.issue for issue in report.issues)
        self.assertIn("emergency-warning", joined)

    def test_visual_prompt_validation_does_not_treat_ear_as_er(self):
        anchor = "modern flat illustration"
        character = "stable caregiver character"
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "fever_triage",
                    "allowed_props": ["water cup", "phone", "blank digital thermometer"],
                    "forbidden_visuals": [],
                },
                "english": {
                    "visual_style_anchor": anchor,
                    "character_sheet": character,
                    "scenes": [
                        {
                            "scene_id": "scene_02",
                            "caption_role": "tension",
                            "medical_message": "A mild fever is common with diagnosed ear infections.",
                            "scene_visual_action": "Caregiver offers water near a resting child.",
                            "safe_props": ["water cup"],
                            "shot_type": "medium_action",
                            "primary_prop": "water cup",
                            "background": "living room",
                            "tts_text": "A mild fever is common with diagnosed ear infections.",
                            "image_prompt": f"{anchor}. {character}. Caregiver offers a water cup near a resting child under a light blanket.",
                        }
                    ],
                },
            }
        )

        joined = " ".join(issue.issue for issue in report.issues)
        self.assertNotIn("emergency-warning", joined)

    def test_plan_guardrail_repair_adds_safe_tokens_before_validation(self):
        anchor = "modern flat illustration"
        character = "stable caregiver character"
        plan = {
            "visual_brief": {
                "content_type": "fever_triage",
                "allowed_props": ["blank digital thermometer", "water cup", "phone", "blank note"],
                "forbidden_visuals": [],
            },
            "korean": {
                "creative_angle": "열 관찰",
                "trend_rationale": "현실 육아 상황",
                "caption_style": "korean_jjal",
                "tts_style": "calm",
                "bgm_prompt": "Bright bouncy indie-pop with soft claps",
                "bgm_config": {"density": 0.62, "brightness": 0.9},
                "visual_style_anchor": anchor,
                "character_sheet": character,
                "scenes": [
                    {"caption_role": "hook"},
                    {"caption_role": "tension"},
                    {
                        "scene_id": "ko_scene_3",
                        "caption_role": "insight",
                        "medical_message": "38도 이상 열이 지속되면 검사 고려.",
                        "scene_visual_action": "Caregiver writes a note.",
                        "safe_props": ["blank note"],
                        "shot_type": "tabletop_action",
                        "primary_prop": "blank note",
                        "background": "coffee table",
                        "top_text": "열 체크",
                        "bottom_text": "해열제로 조절 안 되고 38도 이상 지속될 때 검사해요",
                        "tts_text": "38도 이상 열이 지속되면 검사해요.",
                        "image_prompt": f"{anchor}. {character}. Caregiver writes carefully at a coffee table.",
                    },
                    {"caption_role": "relief"},
                ],
            },
        }

        repaired = _repair_plan_guardrails(plan)

        self.assertIn("Instrumental", repaired["korean"]["bgm_prompt"])
        scene = repaired["korean"]["scenes"][2]
        self.assertLessEqual(len(scene["bottom_text"]), 24)
        self.assertIn("blank note", scene["image_prompt"])
        self.assertIn("blank digital thermometer", scene["image_prompt"])

    def test_visual_prompt_validation_allows_negated_text_logo_and_textures(self):
        anchor = "Clean vector illustration with soft textures"
        character = "stable caregiver character"
        report = deterministic_visual_prompt_report(
            {
                "visual_brief": {
                    "content_type": "fever_triage",
                    "allowed_props": ["blank digital thermometer", "water cup"],
                    "forbidden_visuals": [],
                },
                "english": {
                    "visual_style_anchor": anchor,
                    "character_sheet": character,
                    "scenes": [
                        {
                            "scene_id": "scene_01",
                            "caption_role": "hook",
                            "medical_message": "Monitor fever calmly.",
                            "scene_visual_action": "Caregiver holds a blank thermometer.",
                            "safe_props": ["blank digital thermometer"],
                            "shot_type": "close_up",
                            "primary_prop": "blank digital thermometer",
                            "background": "bedroom",
                            "tts_text": "Monitor fever calmly.",
                            "image_prompt": (
                                f"{anchor}. {character}. Caregiver holds a blank digital thermometer. "
                                "No text, logos, or warning graphics are present."
                            ),
                        }
                    ],
                },
            }
        )

        joined = " ".join(issue.issue for issue in report.issues)
        self.assertNotIn("visible text", joined)
        self.assertNotIn("visible logos", joined)

    def test_campaign_profile_override_deep_merges_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps({
                    "name": "clinic_brand",
                    "languages": {
                        "korean": {
                            "language_note": "Use clinic-approved Korean phrasing."
                        }
                    },
                }),
                encoding="utf-8",
            )

            profile = load_campaign_profile(path)

        self.assertEqual(profile["name"], "clinic_brand")
        self.assertEqual(profile["languages"]["korean"]["language_note"], "Use clinic-approved Korean phrasing.")
        self.assertIn("platforms", profile["languages"]["korean"])
        self.assertIn("english", profile["languages"])


    def test_dashboard_payload_reads_agent_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260603T000000Z_row7"
            run_dir.mkdir()
            (run_dir / "source.json").write_text(json.dumps({"row_number": 7}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
            (run_dir / "agent_trace.json").write_text(
                json.dumps([{"agent": "render_agent", "status": "succeeded", "message": "1 videos"}]),
                encoding="utf-8",
            )

            payload = build_runs_payload(Path(tmp))

        self.assertEqual(payload["runs"][0]["run_id"], "20260603T000000Z_row7")
        self.assertEqual(payload["runs"][0]["row_number"], 7)
        self.assertEqual(payload["runs"][0]["latest_agent"], "render_agent")

    def test_creative_candidate_parser_accepts_flat_candidates_payload(self):
        flat = {"candidates": []}
        for language in ("English", "Korean", "Spanish"):
            for index in range(3):
                flat["candidates"].append(
                    {
                        "id": f"{language.lower()}_{index + 1}",
                        "language": language,
                        "creative_angle": "Parent searches at night",
                        "trend_rationale": "Native parent reality",
                        "hook": "POV: you searched symptoms at 2am",
                        "caption_style": "clean_reels",
                        "visual_style_anchor": "clean 9:16 editorial illustration",
                        "character_sheet": "tired parent in a gray hoodie",
                        "tts_style": "warm and calm",
                        "bgm_prompt": "gentle instrumental educational underscore, no vocals",
                        "bgm_config": {"bpm": 104, "density": 0.5, "brightness": 0.6},
                        "avoid": ["fear bait"],
                        "beats": ["hook", "tension", "insight", "relief"],
                    }
                )

        parsed = _parse_candidate_set(json.dumps(flat))

        self.assertEqual(len(parsed["english"]), 3)
        self.assertEqual(parsed["english"][0]["language"], "english")
        self.assertEqual(parsed["korean"][0]["caption_style"], "clean_reels")

    def test_creative_candidate_parser_accepts_grouped_trend_like_payload(self):
        payload = {}
        for language in ("english", "korean", "spanish"):
            payload[language] = []
            for index in range(3):
                payload[language].append(
                    {
                        "format_name": f"Night parent format {index + 1}",
                        "hook_templates": ["POV: stomach bug or diarrhea?"],
                        "caption_style": "large clean first-frame hook",
                        "visual_style_anchor": "clean faceless parent illustration",
                        "character_sheet": "tired parent in a navy hoodie",
                        "bgm_prompt": "gentle instrumental educational underscore, no vocals",
                        "bgm_config": {
                            "bpm": "104",
                            "density": "0.52",
                            "brightness": "0.61",
                            "guidance": "comforting parent education cue",
                        },
                        "scene_beats": [
                            {"beat_name": "hook", "visual": "Parent checks a phone"},
                            {"beat_name": "tension", "visual": "Parent hesitates"},
                            {"beat_name": "insight", "visual": "Small sips nearby"},
                            {"beat_name": "relief", "visual": "Parent calmer"},
                        ],
                    }
                )

        parsed = _parse_candidate_set(json.dumps(payload))

        self.assertEqual(parsed["spanish"][0]["creative_angle"], "Night parent format 1")
        self.assertEqual(parsed["english"][0]["hook_template"], "POV: stomach bug or diarrhea?")
        self.assertEqual(parsed["korean"][0]["bgm_config"]["guidance"], 4.0)
        self.assertEqual(parsed["english"][0]["scene_beats"], [
            "Parent checks a phone",
            "Parent hesitates",
            "Small sips nearby",
            "Parent calmer",
        ])

    def test_creative_score_parser_accepts_flat_scores_payload(self):
        flat = {"scores": []}
        for language in ("english", "korean", "spanish"):
            for index in range(3):
                flat["scores"].append(
                    {
                        "candidate_id": f"{language}_{index + 1}",
                        "language": language,
                        "hook_strength": 4,
                        "native_fit": 4,
                        "retention": 4,
                        "medical_safety": 5,
                        "not_cringe": 4,
                        "visual_feasibility": 5,
                        "selected": index == 0,
                    }
                )

        parsed = _parse_score_set(json.dumps(flat))

        self.assertEqual(len(parsed["spanish"]), 3)
        self.assertTrue(parsed["spanish"][0]["selected"])

    def test_creative_score_parser_accepts_evaluations_payload(self):
        payload = {"evaluations": []}
        for language in ("english", "korean", "spanish"):
            for index in range(3):
                payload["evaluations"].append(
                    {
                        "candidate_id": f"{language}_{index + 1}",
                        "scores": {
                            "hook_strength": 4,
                            "native_fit": 4,
                            "retention": 4,
                            "medical_safety": 5,
                            "not_cringe": 4,
                            "visual_feasibility": 5,
                        },
                        "winner": index == 1,
                    }
                )

        parsed = _parse_score_set(json.dumps(payload))

        self.assertTrue(parsed["english"][1]["selected"])
        self.assertEqual(parsed["korean"][0]["medical_safety"], 5)

    def test_meme_plan_parser_accepts_scene_alias_payload(self):
        payload = {
            "english": {
                "creative_angle": "The raw beef panic text",
                "scenes": [{"scene_num": 1, "caption_text": "Did my toddler eat raw beef?"}],
            },
            "korean": {
                "creative_angle": "아이 생고기 검색",
                "scenes": [{"scene_num": 1, "caption_text": "생고기 먹었나요?"}],
            },
            "spanish": {
                "creative_angle": "Carne cruda",
                "scenes": [{"scene_num": 1, "caption_text": "Comió carne cruda?"}],
            },
        }
        candidates = {
            lang: [
                {
                    "candidate_id": f"{lang}_1",
                    "hook_template": "POV",
                    "trend_rationale": "parent reality",
                    "visual_style_anchor": "clean 9:16 social illustration",
                    "character_sheet": "tired parent in casual home clothes",
                    "tts_style": "warm",
                    "bgm_prompt": "gentle instrumental, no vocals",
                    "bgm_config": {"density": 0.5, "brightness": 0.6},
                }
            ]
            for lang in ("english", "korean", "spanish")
        }
        scores = {
            lang: [{"candidate_id": f"{lang}_1", "selected": True}]
            for lang in ("english", "korean", "spanish")
        }

        parsed = _parse_meme_plan(json.dumps(payload), topic="raw beef", candidates=candidates, scores=scores)

        self.assertEqual(parsed["source_topic"], "raw beef")
        self.assertEqual(parsed["english"]["language"], "english")
        self.assertEqual(parsed["english"]["scenes"][0]["scene_id"], "scene_01")
        self.assertEqual(parsed["english"]["scenes"][0]["meme_format"], "pov")

    def test_graph_happy_path_reaches_finalize_with_mock_nodes(self):
        if graph_module.StateGraph is None:
            self.skipTest("langgraph is not installed")

        originals = {}

        def fake_node(name):
            def node(state):
                updated = dict(state)
                updated["agent_trace"] = list(updated.get("agent_trace", [])) + [{"agent": name}]
                return updated
            return node

        patches = {
            "load_source_node": fake_node("load_source"),
            "grounding_agent_node": fake_node("grounding_agent"),
            "script_writer_agent_node": fake_node("script_writer_agent"),
            "persist_script_artifacts_node": fake_node("persist_script_artifacts"),
            "meme_planner_agent_node": fake_node("meme_planner_agent"),
            "image_generation_agent_node": fake_node("image_generation_agent"),
            "tts_agent_node": fake_node("tts_agent"),
            "bgm_agent_node": fake_node("bgm_agent"),
            "render_agent_node": fake_node("render_agent"),
            "finalize_artifacts_node": fake_node("finalize_artifacts"),
            "fail_closed_node": fake_node("fail_closed"),
        }
        try:
            for name, replacement in patches.items():
                originals[name] = getattr(graph_module, name)
                setattr(graph_module, name, replacement)
            result = graph_module.build_pipeline_graph().invoke({"agent_trace": []})
        finally:
            for name, original in originals.items():
                setattr(graph_module, name, original)

        agents = [item["agent"] for item in result["agent_trace"]]
        self.assertIn("finalize_artifacts", agents)
        self.assertNotIn("fail_closed", agents)


if __name__ == "__main__":
    unittest.main()
