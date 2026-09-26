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

    role_id: str
    sources: tuple[int, ...]
    field: str
    value: str

    @property
    def claim(self) -> str:
        if self.field.lower() in {"observation", "claim", "evidence"}:
            return self.value
        return f"{self.field}: {self.value}"


@dataclass(frozen=True)
class EvidenceRole:
    """One answer-free operand role fixed from the question alone."""

    role_id: str
    description: str


@dataclass(frozen=True)
class EvidenceSpecification:
    """Question-only operand roles and the fixed composition operator."""

    roles: tuple[EvidenceRole, ...]
    operator: str
    operator_dag: tuple[str, ...] = ()


@dataclass(frozen=True)
class CertificationConfig:
    """Configuration for exact source-grouped necessity estimation."""

    epsilon: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")


@dataclass(frozen=True)
class VerificationConfig:
    """Natural zero boundaries for Bridge-only response verification."""

    gamma: float = 0.0
    eta: float = 0.0


@dataclass(frozen=True)
class FlowConfig:
    max_atoms: int = 5
    specification_max_new_tokens: int = 256
    proposal_max_new_tokens: int = 320
    composition_max_new_tokens: int = 128
    answer_max_new_tokens: int = 96
    role_isolated_proposal: bool = True

    def __post_init__(self) -> None:
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        for name in (
            "specification_max_new_tokens",
            "proposal_max_new_tokens",
            "composition_max_new_tokens",
            "answer_max_new_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


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

    # Specify_theta(Q) is completed before the model sees any image or response.
    # The resulting roles and operator remain fixed for the rest of the run.
    backend.reset_multimodal_state()
    specification = specify_evidence_structure(
        backend,
        question,
        max_roles=flow.max_atoms,
        max_new_tokens=flow.specification_max_new_tokens,
    )
    provisional = propose_bridge(
        backend,
        question,
        image_paths,
        specification,
        max_atoms=flow.max_atoms,
        max_new_tokens=flow.proposal_max_new_tokens,
        role_isolated=flow.role_isolated_proposal,
    )
    certified_atoms = [
        certify_atom(atom, question, image_paths, backend, certification)
        for atom in provisional
    ]

    factual_bridge = render_bridge(certified_atoms, null=False)
    null_bridge = render_bridge(certified_atoms, null=True)

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

    # Constructed hard reset: every composition, generation, and verification
    # call below is a new text-only computation with no image-derived state.
    backend.reset_multimodal_state()

    factual_composition = compose_bridge(
        backend,
        question,
        specification,
        certified_atoms,
        factual_bridge,
        null=False,
        max_new_tokens=flow.composition_max_new_tokens,
    )
    null_composition = compose_bridge(
        backend,
        question,
        specification,
        certified_atoms,
        null_bridge,
        null=True,
        max_new_tokens=flow.composition_max_new_tokens,
    )

    if not candidates:
        generated = backend.generate_text(
            build_generation_prompt(
                question,
                specification,
                factual_bridge,
                factual_composition,
            ),
            max_new_tokens=flow.answer_max_new_tokens,
        ).strip()
        hypotheses = [generated] if generated else []

    response_scores = [
        verify_response(
            backend,
            question,
            factual_bridge,
            null_bridge,
            factual_composition,
            null_composition,
            hypothesis,
            verification,
        )
        for hypothesis in hypotheses
    ]
    attach_candidate_metadata(response_scores, candidates)
    selected, selected_via = select_verified_response(response_scores)

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
        "evidence_specification": asdict(specification),
        "provisional_bridge": [asdict(atom) | {"claim": atom.claim} for atom in provisional],
        "certified_bridge": certified_atoms,
        "factual_bridge_text": factual_bridge,
        "null_bridge_text": null_bridge,
        "factual_composition": factual_composition,
        "null_composition": null_composition,
        "response_scores": response_scores,
        "source_metrics": source_metrics,
        "hard_reset": {
            "images_removed": True,
            "visual_tokens_removed": True,
            "multimodal_hidden_states_removed": True,
            "kv_cache_reused": False,
        },
        "method": "care_flow_specify_certify_compose",
        "certification_config": asdict(certification),
        "verification_config": asdict(verification),
        "flow_config": asdict(flow),
        "elapsed_sec": round(time.time() - started, 3),
    }


def select_verified_response(
    response_scores: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Return the valid response with maximum factual Bridge support."""

    valid = [score for score in response_scores if score.get("accepted")]
    if valid:
        return max(valid, key=lambda item: (item["r1"], item["response"])), "verified"
    return None, "abstained"


def apply_decision_policy(
    record: dict[str, Any],
    verification: VerificationConfig,
) -> dict[str, Any]:
    """Reapply the factual/null acceptance boundary without model inference."""

    import copy

    out = copy.deepcopy(record)
    response_scores = out.get("response_scores", [])
    for score in response_scores:
        score["accepted"] = (
            float(score["r1"]) > verification.gamma
            and float(score["delta_fd"]) > verification.eta
        )
    selected, selected_via = select_verified_response(response_scores)
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
    return out


def specify_evidence_structure(
    backend: Any,
    question: str,
    max_roles: int,
    max_new_tokens: int,
) -> EvidenceSpecification:
    """Run answer-free Specify_theta(Q) without images or candidate responses."""

    prompt = build_specification_prompt(question, max_roles)
    raw = backend.generate_text(prompt, max_new_tokens=max_new_tokens)
    specification = parse_evidence_specification(raw, max_roles=max_roles)
    if specification is None:
        retry = (
            prompt
            + "\n\nThe previous output was invalid. Return one JSON object with non-empty "
            "roles and a non-empty operator. Do not include evidence values or an answer."
        )
        raw = backend.generate_text(retry, max_new_tokens=max_new_tokens)
        specification = parse_evidence_specification(raw, max_roles=max_roles)
    if specification is None:
        raise ValueError("Specify_theta(Q) did not produce a valid evidence specification")
    return specification


def parse_evidence_specification(
    raw: str, max_roles: int
) -> EvidenceSpecification | None:
    payload = extract_json_payload(raw)
    if not isinstance(payload, dict):
        return None

    raw_roles = payload.get("roles", payload.get("required_roles", []))
    if not isinstance(raw_roles, list):
        return None
    roles: list[EvidenceRole] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_roles, start=1):
        if isinstance(item, str):
            role_id = f"r{index}"
            description = clean_scalar(item)
        elif isinstance(item, dict):
            role_id = clean_scalar(
                item.get("id", item.get("role_id", item.get("name", f"r{index}")))
            )
            description = clean_scalar(
                item.get(
                    "description",
                    item.get("role", item.get("required_evidence", item.get("field", ""))),
                )
            )
        else:
            continue
        role_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", role_id).strip("_") or f"r{index}"
        if not description or role_id in seen:
            continue
        seen.add(role_id)
        roles.append(EvidenceRole(role_id=role_id, description=description))
        if len(roles) >= max_roles:
            break

    raw_operator = payload.get("operator", payload.get("composition", ""))
    raw_dag = payload.get("operator_dag", payload.get("dag", []))
    if isinstance(raw_operator, dict):
        operator_name = clean_scalar(raw_operator.get("name", raw_operator.get("type", "")))
        operator_description = clean_scalar(
            raw_operator.get("description", raw_operator.get("instruction", ""))
        )
        operator = ": ".join(
            part for part in (operator_name, operator_description) if part
        )
        if not raw_dag:
            raw_dag = raw_operator.get("dag", raw_operator.get("steps", []))
    else:
        operator = clean_scalar(raw_operator)

    dag: list[str] = []
    if isinstance(raw_dag, list):
        dag = [clean_scalar(step) for step in raw_dag]
    elif raw_dag:
        dag = [clean_scalar(raw_dag)]
    dag = [step for step in dag if step]
    if not roles or not operator:
        return None
    return EvidenceSpecification(tuple(roles), operator, tuple(dag))


def propose_bridge(
    backend: Any,
    question: str,
    image_paths: Sequence[Path],
    specification: EvidenceSpecification,
    max_atoms: int,
    max_new_tokens: int,
    role_isolated: bool = True,
) -> list[EvidenceAtom]:
    """Fill only the roles fixed by Specify_theta(Q), without composing them."""

    roles = list(specification.roles[:max_atoms])
    if role_isolated:
        atoms: list[EvidenceAtom] = []
        for role in roles:
            prompt = build_role_proposal_prompt(
                question, specification, role, len(image_paths)
            )
            raw = backend.generate_multimodal(
                prompt, list(image_paths), max_new_tokens=max_new_tokens
            )
            parsed = parse_bridge_proposal(
                raw, num_images=len(image_paths), max_atoms=1
            )
            if parsed:
                atom = parsed[0]
                atoms.append(
                    EvidenceAtom(role.role_id, atom.sources, role.description, atom.value)
                )
            else:
                atoms.append(EvidenceAtom(role.role_id, (), role.description, "unknown"))
        return atoms

    prompt = build_proposal_prompt(question, specification, len(image_paths), max_atoms)
    raw = backend.generate_multimodal(prompt, list(image_paths), max_new_tokens=max_new_tokens)
    parsed = parse_bridge_proposal(raw, num_images=len(image_paths), max_atoms=max_atoms)
    if not parsed:
        retry_prompt = (
            prompt
            + "\n\nYour previous format was invalid. Emit one evidence object for every "
            "specified role with its exact role_id, a non-empty sources list, and a value. "
            "Use only image numbers from 1 through "
            + str(len(image_paths))
            + ". Return the corrected JSON object now."
        )
        raw = backend.generate_multimodal(
            retry_prompt, list(image_paths), max_new_tokens=max_new_tokens
        )
        parsed = parse_bridge_proposal(raw, num_images=len(image_paths), max_atoms=max_atoms)
    return align_atoms_to_roles(parsed, roles)


def align_atoms_to_roles(
    atoms: Sequence[EvidenceAtom], roles: Sequence[EvidenceRole]
) -> list[EvidenceAtom]:
    """Preserve every specified role; missing proposals become unknown operands."""

    by_role = {atom.role_id: atom for atom in atoms if atom.role_id}
    aligned: list[EvidenceAtom] = []
    for index, role in enumerate(roles):
        atom = by_role.get(role.role_id)
        if atom is None and index < len(atoms) and not atoms[index].role_id:
            atom = atoms[index]
        if atom is None:
            aligned.append(EvidenceAtom(role.role_id, (), role.description, "unknown"))
        else:
            aligned.append(
                EvidenceAtom(role.role_id, atom.sources, role.description, atom.value)
            )
    return aligned


def certify_atom(
    atom: EvidenceAtom,
    question: str,
    image_paths: Sequence[Path],
    backend: Any,
    config: CertificationConfig,
) -> dict[str, Any]:
    """Estimate N_jk and apply the paper's exact declared-source margin."""

    num_images = len(image_paths)
    declared = {source - 1 for source in atom.sources if 1 <= source <= num_images}
    if not declared or atom.value.casefold() in {"unknown", "unavailable"}:
        return {
            "role_id": atom.role_id,
            "sources": list(atom.sources),
            "field": atom.field,
            "proposed_value": atom.value,
            "value": "unknown",
            "claim": atom.claim,
            "certified": False,
            "baseline_logprob": None,
            "perturbed_logprobs": [],
            "necessity": [],
            "declared_min_necessity": None,
            "strongest_competing_necessity": None,
            "source_margin": None,
        }

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

    declared_min = min(necessities[index] for index in declared)
    competing = [
        necessities[index] for index in range(num_images) if index not in declared
    ]
    strongest_competing = max([0.0, *competing])
    source_margin = declared_min - strongest_competing
    certified = source_margin > 0.0
    return {
        "role_id": atom.role_id,
        "sources": list(atom.sources),
        "field": atom.field,
        "proposed_value": atom.value,
        "value": atom.value if certified else "unknown",
        "claim": atom.claim,
        "certified": certified,
        "baseline_logprob": baseline,
        "perturbed_logprobs": perturbed_scores,
        "necessity": necessities,
        "declared_min_necessity": declared_min,
        "strongest_competing_necessity": strongest_competing,
        "source_margin": source_margin,
    }


def compose_bridge(
    backend: Any,
    question: str,
    specification: EvidenceSpecification,
    certified_atoms: Sequence[dict[str, Any]],
    bridge: str,
    null: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Execute the fixed operator after reset with unknown propagation."""

    unresolved_roles = [
        str(atom.get("role_id", atom.get("field", "unknown_role")))
        for atom in certified_atoms
        if null or str(atom.get("value", "unknown")).casefold() in {"unknown", "unavailable"}
    ]
    if unresolved_roles:
        return {
            "status": "unresolved",
            "result": "unresolved",
            "unresolved_roles": unresolved_roles,
            "raw": None,
        }

    prompt = build_composition_prompt(question, specification, bridge)
    raw = backend.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
    payload = extract_json_payload(raw)
    if isinstance(payload, dict):
        status = clean_scalar(payload.get("status", "resolved")).casefold() or "resolved"
        result = clean_scalar(
            payload.get("result", payload.get("composition", payload.get("value", "")))
        )
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            steps = [steps] if steps else []
        return {
            "status": status,
            "result": result or "unresolved",
            "steps": [clean_scalar(step) for step in steps if clean_scalar(step)],
            "unresolved_roles": [],
            "raw": raw,
        }
    return {
        "status": "resolved" if raw else "unresolved",
        "result": raw or "unresolved",
        "steps": [],
        "unresolved_roles": [],
        "raw": raw,
    }


def verify_response(
    backend: Any,
    question: str,
    factual_bridge: str,
    null_bridge: str,
    factual_composition: dict[str, Any],
    null_composition: dict[str, Any],
    response: str,
    config: VerificationConfig,
) -> dict[str, Any]:
    factual_scores = backend.score_text_candidates(
        build_verification_prompt(
            question, factual_bridge, factual_composition, response
        ),
        ["Yes", "No"],
    )
    null_scores = backend.score_text_candidates(
        build_verification_prompt(question, null_bridge, null_composition, response),
        ["Yes", "No"],
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


def build_specification_prompt(question: str, max_roles: int) -> str:
    return f"""Specify the evidence structure required by the question before seeing any image or candidate answer.
Question: {question}

Define at most {max_roles} answer-free operand roles and one fixed composition operator.
- A role states what evidence must be obtained, never its value.
- Split cross-image requirements into separate roles whenever possible.
- The operator states exactly how the filled roles must be combined.
- For multi-step reasoning, list the operator DAG as ordered dependency steps.
- Do not infer visual facts, mention an option, or give an answer.

Return JSON only:
{{"roles":[{{"id":"r1","description":"attribute of the object in Image 1"}}],"operator":{{"name":"compare","description":"compare r1 with r2","dag":["read r1 and r2","compare them"]}}}}"""


def render_specification(specification: EvidenceSpecification) -> str:
    lines = ["Fixed evidence specification:"]
    for role in specification.roles:
        lines.append(f"- {role.role_id}: {role.description}")
    lines.append(f"Fixed operator: {specification.operator}")
    if specification.operator_dag:
        lines.append("Operator DAG:")
        lines.extend(
            f"  {index}. {step}"
            for index, step in enumerate(specification.operator_dag, start=1)
        )
    return "\n".join(lines)


def build_proposal_prompt(
    question: str,
    specification: EvidenceSpecification,
    num_images: int,
    max_atoms: int,
) -> str:
    return f"""Fill the already fixed evidence roles for a multi-image question.
Question: {question}
There are {num_images} images, numbered Image 1 through Image {num_images}.

{render_specification(specification)}

Produce exactly one atomic operand for each listed role, up to {max_atoms} roles.
- Copy each role_id exactly and provide the exact image source set supporting its value.
- Do not add, remove, rename, or reinterpret roles.
- Do not execute or replace the fixed operator.
- Do not state a final answer, option letter, candidate choice, or answer recommendation.
- Do not use captions, metadata, or outside knowledge.

Return JSON only in this form:
{{"evidence":[{{"role_id":"r1","sources":[1],"value":"brown"}}]}}"""


def build_role_proposal_prompt(
    question: str,
    specification: EvidenceSpecification,
    role: EvidenceRole,
    num_images: int,
) -> str:
    return f"""Fill one role from an evidence structure fixed before image inspection.
Question: {question}
There are {num_images} images, numbered Image 1 through Image {num_images}.

{render_specification(specification)}

Fill only role {role.role_id}: {role.description}
- Inspect pixels and report the exact supporting image source set.
- Do not fill any other role or execute the operator.
- Do not state an answer, option, or recommendation.
- Use "unknown" only when the required operand cannot be identified.

Return JSON only:
{{"evidence":[{{"role_id":"{role.role_id}","sources":[1],"value":"visible fact"}}]}}"""


def build_atom_scoring_prompt(question: str, atom: EvidenceAtom) -> str:
    sources = ", ".join(f"Image {source}" for source in atom.sources)
    return f"""Complete one pre-response CARE-Bridge atom from the images.
Question: {question}
Required role: {atom.role_id}: {atom.field}
Declared visual source: {sources}
Write only the field value supported by the declared source.
Value:"""


def build_baseline_prompt(question: str) -> str:
    return (
        f"Answer the multi-image question using the images.\nQuestion: {question}\n"
        "Complete answer:"
    )


def build_composition_prompt(
    question: str, specification: EvidenceSpecification, bridge: str
) -> str:
    return f"""Execute the fixed operator using only the supplied role values. The original images are unavailable.
Question: {question}

{render_specification(specification)}

{bridge}

- Do not retrieve, infer, or substitute a new operand.
- Follow the fixed operator and its DAG exactly.
- If any required role is unknown or unavailable, return status "unresolved".

Return JSON only:
{{"status":"resolved","steps":["operator step"],"result":"composition result"}}"""


def render_composition(composition: dict[str, Any]) -> str:
    lines = [f"Composition status: {composition.get('status', 'unresolved')}"]
    steps = composition.get("steps") or []
    if steps:
        lines.append("Composition steps:")
        lines.extend(f"- {step}" for step in steps)
    lines.append(f"Composition result: {composition.get('result', 'unresolved')}")
    unresolved = composition.get("unresolved_roles") or []
    if unresolved:
        lines.append("Unresolved roles: " + ", ".join(map(str, unresolved)))
    return "\n".join(lines)


def build_generation_prompt(
    question: str,
    specification: EvidenceSpecification,
    bridge: str,
    composition: dict[str, Any],
) -> str:
    return f"""Answer the question using only the fixed evidence computation below. The original images are unavailable.
If required evidence is unknown, say that the answer cannot be determined.

Question: {question}

{render_specification(specification)}

{bridge}

{render_composition(composition)}

Complete answer:"""


def build_verification_prompt(
    question: str,
    bridge: str,
    composition: dict[str, Any],
    response: str,
) -> str:
    return f"""The original images are unavailable. Judge only from the source-indexed CARE-Bridge.
A response is supported only if every claim uses the supplied operands and agrees with the explicit composition. Unknown, unavailable, or unresolved evidence cannot support a response.

Question: {question}

{bridge}

{render_composition(composition)}

Complete response: {response}

Is the complete response fully and correctly supported by the CARE-Bridge? Answer Yes or No.
Verdict:"""


def render_bridge(
    certified_atoms: Sequence[dict[str, Any]],
    null: bool,
) -> str:
    lines = ["CARE-Bridge:"]
    for atom in certified_atoms:
        labels = ", ".join(f"Image {source}" for source in atom["sources"])
        labels = labels or "no certified source"
        value = "unavailable" if null else str(atom.get("value", "unknown"))
        lines.append(
            f"- {atom.get('role_id', 'role')} [{labels}] {atom['field']}: {value}"
        )
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
    seen: set[tuple[str, tuple[int, ...], str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        sources = normalize_sources(
            item.get("sources", item.get("source", item.get("images", item.get("image"))))
        )
        sources = tuple(source for source in sources if 1 <= source <= num_images)
        if not sources:
            continue
        role_id = clean_scalar(item.get("role_id", item.get("role", "")))
        field = clean_scalar(
            item.get("field", item.get("attribute", item.get("slot", "observation")))
        ) or "observation"
        value = clean_scalar(
            item.get("value", item.get("claim", item.get("fact", item.get("observation", ""))))
        )
        if not value:
            continue
        atom = EvidenceAtom(role_id, tuple(sorted(set(sources))), field, value)
        key = (atom.role_id, atom.sources, atom.field.casefold(), atom.value.casefold())
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
        atoms.append(EvidenceAtom("", tuple(sorted(set(sources))), field, value))
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
