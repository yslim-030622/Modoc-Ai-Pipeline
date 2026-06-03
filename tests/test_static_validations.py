from pathlib import Path
import unittest

from modoc_pipeline.excel_source import load_qna_sources
from modoc_pipeline.grounding import generate_grounding_report
from modoc_pipeline.quality import deterministic_quality_report
from modoc_pipeline.schemas import GroundingReport, ScriptPackage


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


if __name__ == "__main__":
    unittest.main()
