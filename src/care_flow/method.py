from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .text import exact_or_contains


IMAGE_UNIT_RE = re.compile(r"I(?:mage\s*)?(\d+)", re.IGNORECASE)
INTEGER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class EvidenceAtom:
    """One source-indexed atomic field in the provisional CARE-Bridge."""

    sources: tuple[int, ...]
    field: str
    value: str

    @property
    def claim(self) -> str:
        if self.field.lower() in {"observation", "claim", "evidence"}:
            return self.value
        return f"{self.field}: {self.value}"


@dataclass(frozen=True)
class CertificationConfig:
    """Paper thresholds for the X -> M causal certification leg."""

    epsilon: float = 0.9
    kappa: float = 0.0
    rho: float = 0.3
    kappa_min: float = 0.0
    epsilon0: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if self.epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be positive")


@dataclass(frozen=True)
class VerificationConfig:
    """Paper thresholds for selective Bridge-only response verification."""

    gamma: float = 1.0
    eta: float = 6.0


@dataclass(frozen=True)
class FlowConfig:
    max_atoms: int = 5
    proposal_max_new_tokens: int = 320
    answer_max_new_tokens: int = 96
    retain_initial_on_rejection: bool = True
    source_isolated_proposal: bool = True
    retain_tentative_atoms: bool = True
    residual_rerank_weight: float = 1.5
    factual_support_weight: float = 2.0 / 3.0
    residual_rerank_margin: float = 0.4
    baseline_margin_cap: float = 0.75

    def __post_init__(self) -> None:
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        if self.residual_rerank_weight < 0.0:
            raise ValueError("residual_rerank_weight must be non-negative")
        if not 0.0 <= self.factual_support_weight <= 1.0:
            raise ValueError("factual_support_weight must lie in [0, 1]")
        if not 0.0 <= self.residual_rerank_margin <= 1.0:
            raise ValueError("residual_rerank_margin must lie in [0, 1]")
        if self.baseline_margin_cap < 0.0:
            raise ValueError("baseline_margin_cap must be non-negative")


def select_rows(
    rows: list[dict[str, Any]],
    splits: set[str] | None = None,
    source_filter: set[str] | None = None,
    limit: int = 0,
    offset: int = 0,
    sample_size: int = 0,
    sample_seed: int = 17,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if splits and row.get("split") not in splits:
            continue
        if source_filter and not (set(row.get("source_mix", [])) & source_filter):
            continue
        selected.append(row)
    if sample_size > 0:
        import random

        rng = random.Random(sample_seed)
        selected = rng.sample(selected, min(sample_size, len(selected)))
    if offset:
        selected = selected[offset:]
    if limit > 0:
        selected = selected[:limit]
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    if num_shards > 1:
        selected = selected[shard_index::num_shards]
    return selected


def run_care_flow_on_row(
    row: dict[str, Any],
    backend: Any,
    dataset_dir: Path,
    certification: CertificationConfig,
    verification: VerificationConfig,
    flow: FlowConfig,
) -> dict[str, Any]:
    """Run Algorithm 1 from the paper on one CARE-Bench instance.

    No benchmark annotation is passed to proposal, certification, generation,
    or verification. Gold fields are consulted only after inference to compute
    diagnostics.
    """

    started = time.time()
    question = str(row.get("question", "")).strip()
    image_paths = resolve_image_paths(row, dataset_dir)
    candidates = list(row.get("candidates") or [])

    provisional = propose_bridge(
        backend,
        question,
        image_paths,
        max_atoms=flow.max_atoms,
        max_new_tokens=flow.proposal_max_new_tokens,
        source_isolated=flow.source_isolated_proposal,
    )
    certified_atoms = [
        certify_atom(atom, question, image_paths, backend, certification)
        for atom in provisional
    ]

    factual_bridge = render_bridge(
        certified_atoms,
        null=False,
        retain_tentative=flow.retain_tentative_atoms,
    )
    null_bridge = render_bridge(
        certified_atoms,
        null=True,
        retain_tentative=flow.retain_tentative_atoms,
    )

    baseline_scores: dict[str, float] = {}
    baseline_generated: str | None = None
    if candidates:
        candidate_texts = [str(candidate.get("text", "")).strip() for candidate in candidates]
        baseline_scores = backend.score_multimodal_candidates(
            build_baseline_prompt(question), image_paths, candidate_texts
        )
        scored_baseline_candidate = max(
            candidates,
            key=lambda candidate: (
                baseline_scores[str(candidate.get("text", "")).strip()],
                str(candidate.get("candidate_id", "")),
            ),
        )
        if "_care_initial_candidate_id" in row:
            external_id = row.get("_care_initial_candidate_id")
            baseline_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.get("candidate_id") == external_id
                ),
                None,
            )
            baseline_source = "external_full_release_vanilla"
        else:
            baseline_candidate = scored_baseline_candidate
            baseline_source = "candidate_commitment"
        hypotheses = candidate_texts
    else:
        baseline_candidate = None
        if "_care_initial_answer" in row:
            baseline_source = "external_full_release_vanilla"
            baseline_generated = str(row.get("_care_initial_answer") or "").strip()
        else:
            baseline_source = "generated"
            baseline_generated = backend.generate_multimodal(
                build_baseline_prompt(question),
                image_paths,
                max_new_tokens=flow.answer_max_new_tokens,
            ).strip()

    # Constructed hard reset: every verification and structured-decoding call
    # below receives no image-derived state. In forced-choice mode the initial
    # hypothesis may be retained when CARE has no supported correction;
    # selective-only mode instead abstains and preserves strict mediation.
    backend.reset_multimodal_state()

    if not candidates:
        generated = backend.generate_text(
            build_generation_prompt(question, factual_bridge),
            max_new_tokens=flow.answer_max_new_tokens,
        ).strip()
        hypotheses = [generated] if generated else []

    response_scores = [
        verify_response(
            backend,
            question,
            factual_bridge,
            null_bridge,
            hypothesis,
            verification,
        )
        for hypothesis in hypotheses
    ]
    attach_candidate_metadata(response_scores, candidates)
    structured_candidate_id = structured_bridge_candidate_id(
        certified_atoms,
        response_scores,
        include_tentative=flow.retain_tentative_atoms,
    )
    selected, selected_via = select_verified_response(
        response_scores,
        baseline_candidate.get("candidate_id") if baseline_candidate else None,
        retain_initial_on_rejection=flow.retain_initial_on_rejection,
        structured_candidate_id=structured_candidate_id,
        baseline_scores=baseline_scores,
        residual_rerank_weight=(
            flow.residual_rerank_weight if flow.retain_initial_on_rejection else 0.0
        ),
        factual_support_weight=flow.factual_support_weight,
        residual_rerank_margin=flow.residual_rerank_margin,
        baseline_margin_cap=flow.baseline_margin_cap,
    )

    candidate_by_text = {
        str(candidate.get("text", "")).strip(): candidate for candidate in candidates
    }
    selected_candidate = candidate_by_text.get(selected["response"]) if selected else None
    gold_id = row.get("gold_candidate_id")
    reference_answer = str(row.get("reference_answer") or "").strip() or None
    baseline_id = baseline_candidate.get("candidate_id") if baseline_candidate else None
    selected_id = selected_candidate.get("candidate_id") if selected_candidate else None
    baseline_answer = (
        baseline_candidate.get("text") if baseline_candidate else baseline_generated
    )
    care_answer = selected["response"] if selected else None
    baseline_correct = (
        bool(baseline_candidate and baseline_id == gold_id)
        if candidates
        else bool(reference_answer and exact_or_contains(baseline_answer or "", reference_answer))
    )
    care_correct = (
        bool(selected_candidate and selected_id == gold_id)
        if candidates
        else bool(reference_answer and care_answer and exact_or_contains(care_answer, reference_answer))
    )

    proposed_sources = {source for atom in provisional for source in atom.sources}
    certified_sources = {
        source
        for atom in certified_atoms
        if atom["certified"]
        for source in atom["sources"]
    }
    required_sources = parse_required_sources(row.get("question_evidence_set", []))
    source_metrics = source_set_metrics(proposed_sources, certified_sources, required_sources)

    return {
        "instance_id": row.get("instance_id"),
        "split": row.get("split"),
        "source_mix": row.get("source_mix"),
        "task_type": row.get("task_type"),
        "error_focus": row.get("error_focus"),
        "question": question,
        "num_images": len(image_paths),
        "num_candidates": len(candidates),
        "gold_candidate_id": gold_id,
        "reference_answer": reference_answer,
        "baseline_candidate_id": baseline_id,
        "baseline_answer": baseline_answer,
        "baseline_source": baseline_source,
        "baseline_correct": baseline_correct,
        "baseline_error_label": selected_error_label(baseline_candidate, gold_id),
        "care_candidate_id": selected_id,
        "care_answer": care_answer,
        "care_abstained": selected is None,
        "care_correct": care_correct,
        "care_error_label": selected_error_label(selected_candidate, gold_id),
        "care_selected_via": selected_via,
        "baseline_scores": baseline_scores,
        "provisional_bridge": [asdict(atom) | {"claim": atom.claim} for atom in provisional],
        "certified_bridge": certified_atoms,
        "factual_bridge_text": factual_bridge,
        "null_bridge_text": null_bridge,
        "response_scores": response_scores,
        "source_metrics": source_metrics,
        "hard_reset": {
            "images_removed": True,
            "visual_tokens_removed": True,
            "multimodal_hidden_states_removed": True,
            "kv_cache_reused": False,
        },
        "method": "care_flow_exact_source_value_output",
        "certification_config": asdict(certification),
        "verification_config": asdict(verification),
        "flow_config": asdict(flow),
        "elapsed_sec": round(time.time() - started, 3),
    }


def select_verified_response(
    response_scores: Sequence[dict[str, Any]],
    initial_candidate_id: Any,
    retain_initial_on_rejection: bool,
    structured_candidate_id: Any = None,
    baseline_scores: dict[str, float] | None = None,
    residual_rerank_weight: float = 0.0,
    factual_support_weight: float = 2.0 / 3.0,
    residual_rerank_margin: float = 0.4,
    baseline_margin_cap: float = 0.75,
) -> tuple[dict[str, Any] | None, str]:
    """Apply CARE verification as a conservative correction to a hypothesis."""

    if structured_candidate_id is not None:
        structured = next(
            (
                score
                for score in response_scores
                if score.get("candidate_id") == structured_candidate_id
            ),
            None,
        )
        if structured is not None:
            return structured, "structured_bridge_decoding"
    valid = [score for score in response_scores if score.get("accepted")]
    if valid:
        return max(valid, key=lambda item: (item["r1"], item["response"])), "verified"
    residual = residual_reranked_response(
        response_scores,
        baseline_scores or {},
        residual_rerank_weight=residual_rerank_weight,
        factual_support_weight=factual_support_weight,
        residual_rerank_margin=residual_rerank_margin,
        baseline_margin_cap=baseline_margin_cap,
    )
    if residual is not None:
        return residual, "bridge_residual_reranking"
    if retain_initial_on_rejection and initial_candidate_id is not None:
        initial = next(
            (
                score
                for score in response_scores
                if score.get("candidate_id") == initial_candidate_id
            ),
            None,
        )
        if initial is not None:
            return initial, "initial_hypothesis_retained"
    return None, "abstained"


def residual_reranked_response(
    response_scores: Sequence[dict[str, Any]],
    baseline_scores: dict[str, float],
    residual_rerank_weight: float,
    factual_support_weight: float,
    residual_rerank_margin: float,
    baseline_margin_cap: float,
) -> dict[str, Any] | None:
    """Rerank an initial candidate distribution with relative Bridge support.

    Absolute verifier logits are poorly calibrated across task families. This
    rule keeps the same factual/null Bridge intervention but uses only
    within-question relative support. A margin gate leaves ambiguous cases at
    the initial hypothesis.
    """

    if residual_rerank_weight <= 0.0 or len(response_scores) < 2:
        return None
    responses = [str(score.get("response", "")).strip() for score in response_scores]
    if len(set(responses)) != len(responses) or any(
        response not in baseline_scores for response in responses
    ):
        return None

    raw_baseline = [float(baseline_scores[response]) for response in responses]
    baseline_order = sorted(raw_baseline, reverse=True)
    if baseline_order[0] - baseline_order[1] > baseline_margin_cap:
        return None
    baseline = minmax_normalize(raw_baseline)
    factual = minmax_normalize([float(score["r1"]) for score in response_scores])
    uplift = minmax_normalize(
        [float(score["delta_fd"]) for score in response_scores]
    )
    bridge = [
        factual_support_weight * factual[index]
        + (1.0 - factual_support_weight) * uplift[index]
        for index in range(len(response_scores))
    ]
    bridge_order = sorted(
        range(len(response_scores)),
        key=lambda index: (
            bridge[index],
            str(response_scores[index].get("candidate_id", "")),
        ),
        reverse=True,
    )
    if bridge[bridge_order[0]] - bridge[bridge_order[1]] < residual_rerank_margin:
        return None

    selected_index = max(
        range(len(response_scores)),
        key=lambda index: (
            baseline[index] + residual_rerank_weight * bridge[index],
            bridge[index],
            baseline[index],
            str(response_scores[index].get("candidate_id", "")),
        ),
    )
    return response_scores[selected_index]


def minmax_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    span = max(values) - lower
    if span <= 1e-12:
        return [0.0 for _ in values]
    return [(value - lower) / span for value in values]


def apply_decision_policy(
    record: dict[str, Any],
    verification: VerificationConfig,
    retain_initial_on_rejection: bool,
    flow: FlowConfig | None = None,
) -> dict[str, Any]:
    """Reapply frozen verifier thresholds without rerunning the model."""

    import copy

    out = copy.deepcopy(record)
    response_scores = out.get("response_scores", [])
    for score in response_scores:
        score["accepted"] = (
            float(score["r1"]) > verification.gamma
            and float(score["delta_fd"]) > verification.eta
        )
    if flow is None:
        stored_flow = out.get("flow_config") or {}
        flow = FlowConfig(
            **{
                key: value
                for key, value in stored_flow.items()
                if key in FlowConfig.__dataclass_fields__
            }
        )
    selected, selected_via = select_verified_response(
        response_scores,
        out.get("baseline_candidate_id"),
        retain_initial_on_rejection,
        structured_candidate_id=structured_bridge_candidate_id(
            out.get("certified_bridge", []),
            response_scores,
            include_tentative=bool(
                (out.get("flow_config") or {}).get("retain_tentative_atoms")
            ),
        ),
        baseline_scores=out.get("baseline_scores") or {},
        residual_rerank_weight=(
            flow.residual_rerank_weight if retain_initial_on_rejection else 0.0
        ),
        factual_support_weight=flow.factual_support_weight,
        residual_rerank_margin=flow.residual_rerank_margin,
        baseline_margin_cap=flow.baseline_margin_cap,
    )
    selected_id = selected.get("candidate_id") if selected else None
    gold_id = out.get("gold_candidate_id")
    out.update(
        {
            "care_candidate_id": selected_id,
            "care_answer": selected.get("response") if selected else None,
            "care_abstained": selected is None,
            "care_correct": bool(selected and selected_id == gold_id),
            "care_error_label": (
                None
                if selected is None or selected_id == gold_id
                else selected.get("error_label") or "unknown"
            ),
            "care_selected_via": selected_via,
            "verification_config": asdict(verification),
        }
    )
    flow_config = out.get("flow_config")
    if isinstance(flow_config, dict):
        flow_config["retain_initial_on_rejection"] = retain_initial_on_rejection
        flow_config["residual_rerank_weight"] = flow.residual_rerank_weight
        flow_config["factual_support_weight"] = flow.factual_support_weight
        flow_config["residual_rerank_margin"] = flow.residual_rerank_margin
        flow_config["baseline_margin_cap"] = flow.baseline_margin_cap
    return out


def structured_bridge_candidate_id(
    atoms: Sequence[dict[str, Any]],
    response_scores: Sequence[dict[str, Any]],
    include_tentative: bool,
) -> Any:
    """Compose structured presence atoms after the hard reset."""

    presence: dict[int, bool] = {}
    for atom in atoms:
        if atom.get("field") != "requested cue presence":
            continue
        if not atom.get("certified") and not include_tentative:
            continue
        sources = list(atom.get("sources") or [])
        value = str(
            atom.get("value")
            if atom.get("certified")
            else atom.get("proposed_value")
        ).casefold()
        if len(sources) == 1 and value in {"present", "absent"}:
            presence[int(sources[0])] = value == "present"
    if not presence:
        return None

    by_text = {
        str(score.get("response", "")).strip().casefold(): score
        for score in response_scores
    }
    exact_image_choices = {
        int(match.group(1)): score
        for text, score in by_text.items()
        if (match := re.fullmatch(r"image\s+(\d+)", text))
    }
    positives = [source for source, visible in presence.items() if visible]
    if exact_image_choices and len(exact_image_choices) == len(by_text):
        if len(positives) != 1:
            return None
        selected = exact_image_choices.get(positives[0])
        return selected.get("candidate_id") if selected else None

    if len(presence) == 2:
        if len(positives) == 2:
            target = "both images"
        elif not positives:
            target = "neither image"
        else:
            target = f"image {positives[0]} only"
        selected = by_text.get(target)
        return selected.get("candidate_id") if selected else None
    return None


def propose_bridge(
    backend: Any,
    question: str,
    image_paths: Sequence[Path],
    max_atoms: int,
    max_new_tokens: int,
    source_isolated: bool = True,
) -> list[EvidenceAtom]:
    if source_isolated:
        atoms: list[EvidenceAtom] = []
        for source in proposal_source_slots(question, len(image_paths)):
            if uses_binary_presence_schema(question):
                yes_no = backend.score_multimodal_candidates(
                    build_source_presence_prompt(question, source, len(image_paths)),
                    [image_paths[source - 1]],
                    ["Yes", "No"],
                )
                value = "present" if yes_no["Yes"] > yes_no["No"] else "absent"
                atoms.append(
                    EvidenceAtom((source,), "requested cue presence", value)
                )
                if len(atoms) >= max_atoms:
                    break
                continue
            prompt = build_source_isolated_proposal_prompt(
                question, source, len(image_paths)
            )
            raw = backend.generate_multimodal(
                prompt, [image_paths[source - 1]], max_new_tokens=max_new_tokens
            )
            parsed = parse_bridge_proposal(
                raw, num_images=len(image_paths), max_atoms=1
            )
            if parsed:
                atom = parsed[0]
                atoms.append(EvidenceAtom((source,), atom.field, atom.value))
            if len(atoms) >= max_atoms:
                break
        return atoms

    prompt = build_proposal_prompt(question, len(image_paths), max_atoms)
    raw = backend.generate_multimodal(prompt, list(image_paths), max_new_tokens=max_new_tokens)
    atoms = parse_bridge_proposal(raw, num_images=len(image_paths), max_atoms=max_atoms)
    if not atoms:
        retry_prompt = (
            prompt
            + "\n\nYour previous format was invalid. Emit at least one evidence object with "
            "a non-empty sources list, a non-empty field, and a non-empty value. "
            "Use only image numbers from 1 through "
            + str(len(image_paths))
            + ". Return the corrected JSON object now."
        )
        raw = backend.generate_multimodal(
            retry_prompt, list(image_paths), max_new_tokens=max_new_tokens
        )
        atoms = parse_bridge_proposal(raw, num_images=len(image_paths), max_atoms=max_atoms)
    # An empty proposal is a valid selective outcome: with no visual evidence
    # entering the mediator, the response stage receives an explicitly unknown
    # Bridge and should abstain. It must not fall back to benchmark annotations.
    return atoms


def proposal_source_slots(question: str, num_images: int) -> list[int]:
    """Return question-named sources, or every source for an image search."""

    named: list[int] = []
    for match in IMAGE_UNIT_RE.findall(question):
        source = int(match)
        if 1 <= source <= num_images and source not in named:
            named.append(source)
    return named or list(range(1, num_images + 1))


def certify_atom(
    atom: EvidenceAtom,
    question: str,
    image_paths: Sequence[Path],
    backend: Any,
    config: CertificationConfig,
) -> dict[str, Any]:
    """Estimate N_jk and apply the paper's G/A/multi-source tests."""

    num_images = len(image_paths)
    prompt = build_atom_scoring_prompt(question, atom)
    baseline_gates = [1.0] * num_images
    baseline = backend.score_multimodal_completion(
        prompt,
        list(image_paths),
        atom.value,
        image_gates=baseline_gates,
    )
    necessities: list[float] = []
    perturbed_scores: list[float] = []
    for image_index in range(num_images):
        gates = [1.0] * num_images
        gates[image_index] = 1.0 - config.epsilon
        weakened = backend.score_multimodal_completion(
            prompt,
            list(image_paths),
            atom.value,
            image_gates=gates,
        )
        perturbed_scores.append(weakened)
        necessities.append((baseline - weakened) / config.epsilon)

    declared = {source - 1 for source in atom.sources}
    positive = [max(value, 0.0) for value in necessities]
    grounding = sum(positive[index] for index in declared)
    attribution = grounding / (sum(positive) + config.epsilon0)
    multi_source_ok = len(declared) <= 1 or all(
        necessities[index] > config.kappa_min for index in declared
    )
    certified = grounding > config.kappa and attribution > config.rho and multi_source_ok
    return {
        "sources": list(atom.sources),
        "field": atom.field,
        "proposed_value": atom.value,
        "value": atom.value if certified else "unknown",
        "claim": atom.claim,
        "certified": certified,
        "baseline_logprob": baseline,
        "perturbed_logprobs": perturbed_scores,
        "necessity": necessities,
        "grounding": grounding,
        "attribution": attribution,
        "multi_source_ok": multi_source_ok,
    }


def verify_response(
    backend: Any,
    question: str,
    factual_bridge: str,
    null_bridge: str,
    response: str,
    config: VerificationConfig,
) -> dict[str, Any]:
    factual_scores = backend.score_text_candidates(
        build_verification_prompt(question, factual_bridge, response), ["Yes", "No"]
    )
    null_scores = backend.score_text_candidates(
        build_verification_prompt(question, null_bridge, response), ["Yes", "No"]
    )
    r1 = float(factual_scores["Yes"] - factual_scores["No"])
    r0 = float(null_scores["Yes"] - null_scores["No"])
    delta = r1 - r0
    return {
        "response": response,
        "r1": r1,
        "r0": r0,
        "delta_fd": delta,
        "accepted": r1 > config.gamma and delta > config.eta,
        "factual_yes_no": factual_scores,
        "null_yes_no": null_scores,
    }


def build_proposal_prompt(question: str, num_images: int, max_atoms: int) -> str:
    return f"""You are constructing a pre-response evidence ledger for a multi-image question.
Question: {question}
There are {num_images} images, numbered Image 1 through Image {num_images}.

Before deciding or wording the answer, record at most {max_atoms} question-relevant atomic visual observations.
- Each observation must have the exact image source(s) that visually support it.
- Prefer separate single-image observations. Use multiple sources only for an irreducible cross-view identity or relation.
- Cover every image explicitly named by the question with its own relevant observation before using any remaining slot.
- For a comparison, record the requested attribute separately for both compared images; do not perform the comparison yet.
- The field describes what must be observed; the value contains only the observed fact.
- Do not state a final answer, option letter, candidate choice, or answer recommendation.
- Do not use captions, metadata, or outside knowledge.

Return JSON only in this form:
{{"evidence":[{{"sources":[1],"field":"relevant object color","value":"brown"}}]}}"""


def build_source_isolated_proposal_prompt(
    question: str, source: int, num_images: int
) -> str:
    if uses_binary_presence_schema(question):
        raise ValueError("Binary presence atoms must use likelihood scoring")
    value_instruction = (
        "Record the requested object, attribute, state, or color as a short value."
    )
    example = (
        f'{{"evidence":[{{"sources":[{source}],'
        '"field":"relevant visual evidence","value":"visible observation"}]}'
        "}"
    )
    return f"""You are constructing CARE-Bridge before response formation.
The one shown image is original Image {source} from a {num_images}-image question.
Question: {question}

Record exactly one atomic visual observation from original Image {source} that is needed to answer the question.
- Inspect only the shown pixels. Preserve its original source number {source}.
- {value_instruction}
- Do not compare images or state a final answer, option, or recommendation.
- The field names what is inspected; the value contains only the observation.

Return JSON only:
{example}"""


def uses_binary_presence_schema(question: str) -> bool:
    explicit_count = len({int(match) for match in IMAGE_UNIT_RE.findall(question)})
    lowered = question.casefold()
    return (
        explicit_count != 1
        and "color comparison" not in lowered
        and any(
            cue in lowered
            for cue in (
                " visible",
                "which image",
                "which picture",
                "which view",
                "where ",
                "where can",
                "contains",
                "appear",
                "has the visible cue",
                "shows ",
            )
        )
    )


def build_source_presence_prompt(
    question: str, source: int, num_images: int
) -> str:
    return f"""The shown image is original Image {source} from a {num_images}-image question.
Original question: {question}

Considering only the complete visual cue requested by the question, is that cue visible in original Image {source}? Inspect pixels only. Answer Yes or No.
Verdict:"""


def build_atom_scoring_prompt(question: str, atom: EvidenceAtom) -> str:
    sources = ", ".join(f"Image {source}" for source in atom.sources)
    return f"""Complete one pre-response CARE-Bridge atom from the images.
Question: {question}
Declared visual source: {sources}
Atomic field: {atom.field}
Write only the field value supported by the declared source.
Value:"""


def build_baseline_prompt(question: str) -> str:
    return (
        f"Answer the multi-image question using the images.\nQuestion: {question}\n"
        "Complete answer:"
    )


def build_generation_prompt(question: str, bridge: str) -> str:
    return f"""Answer the question using only the CARE-Bridge below. The original images are unavailable.
If required evidence is unknown, say that the answer cannot be determined.

Question: {question}

{bridge}

Complete answer:"""


def build_verification_prompt(question: str, bridge: str, response: str) -> str:
    return f"""The original images are unavailable. Judge only from the source-indexed CARE-Bridge.
A response is supported only if every claim uses the correct image source, the requested relation is correct, all required evidence is present, and no required field is unknown or unavailable.

Question: {question}

{bridge}

Complete response: {response}

Is the complete response fully and correctly supported by the CARE-Bridge? Answer Yes or No.
Verdict:"""


def render_bridge(
    certified_atoms: Sequence[dict[str, Any]],
    null: bool,
    retain_tentative: bool = False,
) -> str:
    lines = ["CARE-Bridge:"]
    for atom in certified_atoms:
        labels = ", ".join(f"Image {source}" for source in atom["sources"])
        if null:
            value = "unavailable"
        elif atom.get("certified") or not retain_tentative:
            value = str(atom["value"])
        else:
            value = str(atom.get("proposed_value", "unknown"))
        status = "certified" if atom.get("certified") else "tentative"
        suffix = f" [status: {status}]" if retain_tentative else ""
        lines.append(f"- [{labels}] {atom['field']}: {value}{suffix}")
    if len(lines) == 1:
        lines.append("- [no source] evidence: unavailable" if null else "- [no source] evidence: unknown")
    return "\n".join(lines)


def parse_bridge_proposal(raw: str, num_images: int, max_atoms: int) -> list[EvidenceAtom]:
    payload = extract_json_payload(raw)
    items: list[Any] = []
    if isinstance(payload, dict):
        for key in ("evidence", "atoms", "bridge", "observations"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        if not items and any(key in payload for key in ("sources", "source", "images")):
            items = [payload]
    elif isinstance(payload, list):
        items = payload

    atoms: list[EvidenceAtom] = []
    seen: set[tuple[tuple[int, ...], str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        sources = normalize_sources(
            item.get("sources", item.get("source", item.get("images", item.get("image"))))
        )
        sources = tuple(source for source in sources if 1 <= source <= num_images)
        if not sources:
            continue
        field = clean_scalar(
            item.get("field", item.get("attribute", item.get("slot", "observation")))
        ) or "observation"
        value = clean_scalar(
            item.get("value", item.get("claim", item.get("fact", item.get("observation", ""))))
        )
        if not value:
            continue
        atom = EvidenceAtom(tuple(sorted(set(sources))), field, value)
        key = (atom.sources, atom.field.casefold(), atom.value.casefold())
        if key in seen:
            continue
        seen.add(key)
        atoms.append(atom)
        if len(atoms) >= max_atoms:
            break

    if atoms:
        return atoms

    # Strict line fallback for otherwise valid model output such as
    # "Image 2 | object color | brown". Free prose without a source is rejected.
    for line in raw.splitlines():
        match = re.match(
            r"\s*(?:[-*]\s*)?(?:\[)?Images?\s*([\d, &]+)(?:\])?\s*[:|;-]\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        sources = tuple(
            source for source in normalize_sources(match.group(1)) if 1 <= source <= num_images
        )
        parts = [part.strip() for part in match.group(2).split("|") if part.strip()]
        if not sources or not parts:
            continue
        field, value = (parts[0], parts[1]) if len(parts) >= 2 else ("observation", parts[0])
        atoms.append(EvidenceAtom(tuple(sorted(set(sources))), field, value))
        if len(atoms) >= max_atoms:
            break
    return atoms


def extract_json_payload(raw: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[index:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def normalize_sources(value: Any) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            out.extend(normalize_sources(item))
        return tuple(out)
    if isinstance(value, str):
        return tuple(int(match) for match in INTEGER_RE.findall(value))
    return ()


def clean_scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().strip("`\"'")


def parse_required_sources(units: Iterable[Any]) -> set[int]:
    sources: set[int] = set()
    for unit in units:
        sources.update(int(match) for match in IMAGE_UNIT_RE.findall(str(unit)))
        if not IMAGE_UNIT_RE.search(str(unit)):
            sources.update(int(match) for match in re.findall(r"\bI(\d+)\b", str(unit)))
    return sources


def source_set_metrics(
    proposed: set[int], certified: set[int], required: set[int]
) -> dict[str, Any]:
    def recall(values: set[int]) -> float | None:
        return len(values & required) / len(required) if required else None

    def precision(values: set[int]) -> float | None:
        return len(values & required) / len(values) if values else None

    return {
        "required_sources": sorted(required),
        "proposed_sources": sorted(proposed),
        "certified_sources": sorted(certified),
        "proposed_source_recall": recall(proposed),
        "certified_source_recall": recall(certified),
        "proposed_source_precision": precision(proposed),
        "certified_source_precision": precision(certified),
    }


def selected_error_label(candidate: dict[str, Any] | None, gold_id: Any) -> str | None:
    if not candidate or candidate.get("candidate_id") == gold_id:
        return None
    return candidate.get("error_label_if_selected") or "unknown"


def resolve_image_paths(row: dict[str, Any], dataset_dir: Path) -> list[Path]:
    repo_root = Path.cwd()
    paths: list[Path] = []
    images = sorted(row.get("images", []), key=lambda item: int(item.get("display_index", 0)))
    for image in images:
        raw_path = image.get("local_file") or image.get("image_file")
        candidates: list[Path] = []
        if raw_path:
            raw = Path(raw_path)
            candidates.extend([raw, dataset_dir / raw])
            if not raw.is_absolute():
                candidates.append(repo_root / raw)
        uid = str(image.get("image_uid", ""))
        candidates.extend(
            [
                dataset_dir / "coco_images" / f"{uid}.jpg",
                dataset_dir / "coco_images" / f"{uid.lower()}.jpg",
                dataset_dir / "images" / f"{uid}.jpg",
                dataset_dir / "images" / f"{uid.lower()}.jpg",
            ]
        )
        resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not resolve image {uid!r} for instance {row.get('instance_id')!r}"
            )
        paths.append(resolved)
    if not paths:
        raise FileNotFoundError(f"Instance {row.get('instance_id')!r} has no images")
    return paths


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = [record for record in records if not record.get("runtime_error")]
    if not valid_records:
        return {"n": 0, "runtime_errors": len(records)}
    total = len(valid_records)
    accepted = [record for record in valid_records if not record.get("care_abstained", True)]
    baseline_correct = sum(bool(record.get("baseline_correct")) for record in valid_records)
    care_correct = sum(bool(record.get("care_correct")) for record in valid_records)
    corrected = sum(
        not record.get("baseline_correct") and record.get("care_correct")
        for record in valid_records
    )
    harmed = sum(
        record.get("baseline_correct") and not record.get("care_correct")
        for record in valid_records
    )
    accepted_correct = sum(bool(record.get("care_correct")) for record in accepted)

    candidate_scores = [
        (record, score)
        for record in valid_records
        for score in record.get("response_scores", [])
    ]
    invalid_scores = [
        score
        for record, score in candidate_scores
        if response_candidate_id(record, score["response"]) != record.get("gold_candidate_id")
    ]
    gold_scores = [
        score
        for record, score in candidate_scores
        if response_candidate_id(record, score["response"]) == record.get("gold_candidate_id")
    ]
    proposed_atom_count = sum(
        len(record.get("provisional_bridge", [])) for record in valid_records
    )
    certified_atom_count = sum(
        sum(bool(atom.get("certified")) for atom in record.get("certified_bridge", []))
        for record in valid_records
    )
    error_rejection: dict[str, float | None] = {}
    for label in ("WS", "WI", "MI"):
        subset = [
            score
            for record, score in candidate_scores
            if response_error_label(record, score["response"]) == label
        ]
        error_rejection[label] = (
            sum(not score["accepted"] for score in subset) / len(subset) if subset else None
        )

    return {
        "n": total,
        "runtime_errors": len(records) - total,
        "baseline_accuracy": baseline_correct / total,
        "care_accuracy": care_correct / total,
        "accuracy_gain": (care_correct - baseline_correct) / total,
        "corrected": corrected,
        "harmed": harmed,
        "net_corrections": corrected - harmed,
        "accuracy_gain_ci95": paired_bootstrap_gain_ci(valid_records),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(corrected, harmed),
        "care_coverage": len(accepted) / total,
        "care_selective_accuracy": accepted_correct / len(accepted) if accepted else None,
        "care_abstention_rate": 1.0 - len(accepted) / total,
        "gold_acceptance_rate": (
            sum(score["accepted"] for score in gold_scores) / len(gold_scores) if gold_scores else None
        ),
        "invalid_rejection_rate": (
            sum(not score["accepted"] for score in invalid_scores) / len(invalid_scores)
            if invalid_scores
            else None
        ),
        "error_type_rejection": error_rejection,
        "proposed_atom_count": proposed_atom_count,
        "certified_atom_count": certified_atom_count,
        "atom_certification_rate": (
            certified_atom_count / proposed_atom_count if proposed_atom_count else None
        ),
        "mean_proposed_atoms": proposed_atom_count / total,
        "mean_certified_atoms": certified_atom_count / total,
        "mean_proposed_source_recall": optional_mean(
            record.get("source_metrics", {}).get("proposed_source_recall")
            for record in valid_records
        ),
        "mean_certified_source_recall": optional_mean(
            record.get("source_metrics", {}).get("certified_source_recall")
            for record in valid_records
        ),
        "mean_proposed_source_precision": optional_mean(
            record.get("source_metrics", {}).get("proposed_source_precision")
            for record in valid_records
        ),
        "mean_certified_source_precision": optional_mean(
            record.get("source_metrics", {}).get("certified_source_precision")
            for record in valid_records
        ),
        "elapsed_sec": sum(float(record.get("elapsed_sec", 0.0)) for record in valid_records),
    }


def paired_bootstrap_gain_ci(
    records: Sequence[dict[str, Any]], samples: int = 10_000, seed: int = 20260813
) -> list[float]:
    import random

    differences = sorted(
        int(bool(record.get("care_correct")))
        - int(bool(record.get("baseline_correct")))
        for record in records
    )
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(seed)
    gains = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    )
    return [
        gains[int(0.025 * (samples - 1))],
        gains[int(0.975 * (samples - 1))],
    ]


def exact_mcnemar_p(corrected: int, harmed: int) -> float:
    discordant = int(corrected) + int(harmed)
    if discordant == 0:
        return 1.0
    smaller = min(int(corrected), int(harmed))
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def response_candidate_id(record: dict[str, Any], response: str) -> Any:
    for candidate in record.get("benchmark_candidates", []):
        if str(candidate.get("text", "")).strip() == response:
            return candidate.get("candidate_id")
    # Older/newer records need not duplicate candidates; infer selected response
    # where possible and otherwise use the embedded score metadata added below.
    for score in record.get("response_scores", []):
        if score.get("response") == response and "candidate_id" in score:
            return score["candidate_id"]
    if record.get("care_answer") == response:
        return record.get("care_candidate_id")
    return None


def response_error_label(record: dict[str, Any], response: str) -> str | None:
    for score in record.get("response_scores", []):
        if score.get("response") == response:
            return score.get("error_label")
    return None


def attach_candidate_metadata(
    scores: list[dict[str, Any]], candidates: Sequence[dict[str, Any]]
) -> None:
    by_text = {str(candidate.get("text", "")).strip(): candidate for candidate in candidates}
    for score in scores:
        candidate = by_text.get(score["response"])
        if candidate:
            score["candidate_id"] = candidate.get("candidate_id")
            score["is_correct"] = bool(candidate.get("is_correct"))
            score["error_label"] = candidate.get("error_label_if_selected")


def grouped_summaries(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        label = "+".join(map(str, value)) if isinstance(value, list) else str(value)
        groups[label].append(record)
    summaries = []
    for label, group in groups.items():
        summary = summarize_records(group)
        summary["category"] = label
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (-item.get("n", 0), item["category"]))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def optional_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
