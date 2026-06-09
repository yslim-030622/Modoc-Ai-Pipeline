import json
import tempfile
import unittest
from pathlib import Path

from modoc_pipeline.cli import build_parser
from modoc_pipeline.dashboard import build_runs_payload
from modoc_pipeline.meme_planner import (
    _build_candidate_prompt,
    _build_plan_prompt,
    _build_visual_brief_prompt,
    _parse_candidate_set,
    _parse_meme_plan,
    _parse_score_set,
)
import modoc_pipeline.orchestration.graph as graph_module
from modoc_pipeline.orchestration.nodes import route_after_failure
from modoc_pipeline.orchestration.state import PipelineState, ReviewReport
from modoc_pipeline.prompt_profiles import load_campaign_profile
from modoc_pipeline.reviewers import (
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
