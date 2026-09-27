from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from care_flow.method import (
    CertificationConfig,
    EvidenceAtom,
    FlowConfig,
    VerificationConfig,
    certify_atom,
    render_bridge,
    run_care_flow_on_row,
    select_verified_response,
)


class NecessityBackend:
    def __init__(self, necessities: list[float]) -> None:
        self.necessities = necessities

    def score_multimodal_completion(
        self, prompt, image_paths, completion, image_gates
    ) -> float:
        for index, gate in enumerate(image_gates):
            if gate < 1.0:
                epsilon = 1.0 - gate
                return -epsilon * self.necessities[index]
        return 0.0


class StrictFlowBackend(NecessityBackend):
    def __init__(self) -> None:
        super().__init__([2.0, 0.0])
        self.resets = 0

    def reset_multimodal_state(self) -> None:
        self.resets += 1

    def generate_text(self, prompt: str, max_new_tokens: int = 96) -> str:
        if prompt.startswith("Specify the evidence structure"):
            return (
                '{"roles":[{"id":"r1","description":"color in Image 1"}],'
                '"operator":{"name":"return","description":"return r1"}}'
            )
        if prompt.startswith("Execute the fixed operator"):
            return '{"status":"resolved","steps":["return r1"],"result":"red"}'
        raise AssertionError(f"Unexpected text generation prompt: {prompt[:80]}")

    def generate_multimodal(self, prompt, image_paths, max_new_tokens=320) -> str:
        return '{"evidence":[{"role_id":"r1","sources":[1],"value":"red"}]}'

    def score_multimodal_candidates(self, prompt, image_paths, candidates):
        raise AssertionError("Vanilla candidate scores must not enter strict CARE")

    def score_text_candidates(self, prompt, candidates):
        if "CARE-Bridge:" not in prompt:
            raise AssertionError("Verifier must receive the Bridge")
        if (
            "Complete response: red" in prompt
            and "Composition status: resolved" in prompt
        ):
            return {"Yes": 2.0, "No": 0.0}
        return {"Yes": 0.0, "No": 1.0}


class PaperAlignmentTests(unittest.TestCase):
    def test_exact_source_margin_rejects_stronger_competitor(self) -> None:
        atom = EvidenceAtom("r1", (1,), "value", "red")
        result = certify_atom(
            atom,
            "question",
            [Path("i1"), Path("i2")],
            NecessityBackend([1.0, 2.0]),
            CertificationConfig(epsilon=0.9),
        )
        self.assertEqual(result["source_margin"], -1.0)
        self.assertFalse(result["certified"])
        self.assertEqual(result["value"], "unknown")

    def test_exact_source_margin_uses_weakest_declared_source(self) -> None:
        atom = EvidenceAtom("r1", (1, 2), "value", "red")
        result = certify_atom(
            atom,
            "question",
            [Path("i1"), Path("i2"), Path("i3")],
            NecessityBackend([3.0, 2.0, 1.0]),
            CertificationConfig(epsilon=0.9),
        )
        self.assertEqual(result["declared_min_necessity"], 2.0)
        self.assertEqual(result["strongest_competing_necessity"], 1.0)
        self.assertEqual(result["source_margin"], 1.0)
        self.assertTrue(result["certified"])

    def test_failed_atom_never_reenters_factual_bridge(self) -> None:
        bridge = render_bridge(
            [
                {
                    "role_id": "r1",
                    "sources": [1],
                    "field": "color",
                    "proposed_value": "red",
                    "value": "unknown",
                    "certified": False,
                }
            ],
            null=False,
        )
        self.assertIn("unknown", bridge)
        self.assertNotIn("red", bridge)

    def test_zero_boundaries_and_abstention(self) -> None:
        self.assertEqual(VerificationConfig(), VerificationConfig(gamma=0.0, eta=0.0))
        selected, via = select_verified_response(
            [
                {"response": "A", "r1": 5.0, "accepted": False},
                {"response": "B", "r1": 2.0, "accepted": True},
                {"response": "C", "r1": 3.0, "accepted": True},
            ]
        )
        self.assertEqual(selected["response"], "C")
        self.assertEqual(via, "verified")
        selected, via = select_verified_response(
            [{"response": "A", "r1": 5.0, "accepted": False}]
        )
        self.assertIsNone(selected)
        self.assertEqual(via, "abstained")

    def test_external_vanilla_is_metrics_only(self) -> None:
        backend = StrictFlowBackend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("i1.jpg", "i2.jpg"):
                (root / name).touch()
            row = {
                "instance_id": "example",
                "question": "What color is the object in Image 1?",
                "images": [
                    {"display_index": 1, "local_file": "i1.jpg"},
                    {"display_index": 2, "local_file": "i2.jpg"},
                ],
                "candidates": [
                    {"candidate_id": "A", "text": "blue", "is_correct": False},
                    {"candidate_id": "B", "text": "red", "is_correct": True},
                ],
                "gold_candidate_id": "B",
                "_care_initial_candidate_id": "A",
            }
            record = run_care_flow_on_row(
                row,
                backend,
                root,
                CertificationConfig(),
                VerificationConfig(),
                FlowConfig(max_atoms=1),
            )
        self.assertEqual(record["baseline_candidate_id"], "A")
        self.assertEqual(record["care_candidate_id"], "B")
        self.assertEqual(record["care_selected_via"], "verified")
        self.assertTrue(record["care_correct"])
        self.assertGreaterEqual(backend.resets, 2)


if __name__ == "__main__":
    unittest.main()
