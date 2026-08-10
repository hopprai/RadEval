"""Finding-level F1 extraction and evaluation for CT Chest using lv010."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yaml

from .llm_client import call_llm


def strip_version(fid: str) -> str:
    """Strip _v1.0 suffix from finding_ids."""
    return re.sub(r"_v\d+\.\d+$", "", fid)


def load_definitions(path: str) -> list[dict]:
    """Load definitions from JSONL."""
    defs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                defs.append(json.loads(line))
    return defs


def build_catalog(defs: list[dict]) -> str:
    """Build the findings catalog string for the system prompt."""
    blocks = []
    for d in defs:
        fid = strip_version(d["finding_id"])
        name = d.get("finding") or fid
        sig = d.get("clinical_significance") or "-"
        defn = (d.get("clinical_definition") or "").strip()
        aliases = d.get("aliases") or []
        parts = [f"### {fid} — {name}  (clinical_significance: {sig})"]
        if defn:
            parts.append(f"Definition: {defn}")
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def parse_findings(raw: str) -> list[dict]:
    """Parse findings from LLM response — handles JSON, JSONL, YAML, or list formats."""
    raw = raw.strip()
    if not raw:
        return []

    # Strip code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:(-1 if lines[-1].strip() == "```" else len(lines))])
        raw = raw.strip()

    # Try JSON (wrapped object or array)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed.get("findings", [])
        elif isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try JSONL (one object per line)
    try:
        findings = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                findings.append(json.loads(line))
        if findings:
            return findings
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return parsed.get("findings", [])
    except yaml.YAMLError:
        pass

    # Fallback: empty
    return []


def extract_present_findings(
    system_prompt: str,
    report: str,
    model: str,
    valid_findings: set[str],
) -> tuple[set[str], float]:
    """Extract findings with certainty='present' from a single report."""
    try:
        raw, cost = call_llm(
            prompt=f"## Report\n\n{report}\n",
            system_prompt=system_prompt,
            model_name=model,
            temperature=0.0,
            max_tokens=8000,
        )
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return set(), 0.0

    findings = parse_findings(raw)

    # Filter to certainty='present', strip version suffixes, and validate against catalog
    present = set()
    for f in findings:
        if not isinstance(f, dict):
            continue
        if f.get("certainty") == "present":
            finding_id = f.get("finding", "")
            if finding_id:
                normalized = strip_version(finding_id)
                # Only include if it's a valid catalog ID
                if normalized in valid_findings:
                    present.add(normalized)

    return present, cost


def compute_metrics(ref_set: set[str], hyp_set: set[str]) -> dict[str, float]:
    """Compute precision, recall, F1 from two sets."""
    tp = len(ref_set & hyp_set)
    fp = len(hyp_set - ref_set)
    fn = len(ref_set - hyp_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def compute_classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Compute F1, sensitivity, specificity, PPV, NPV from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def process_study(
    ref: str,
    hyp: str,
    study_id: str,
    system_prompt: str,
    model: str,
    valid_findings: set[str],
) -> dict[str, Any]:
    """Process a single study — extract findings from ref and hyp, compute metrics."""
    ref_present, c1 = extract_present_findings(system_prompt, ref, model, valid_findings)
    hyp_present, c2 = extract_present_findings(system_prompt, hyp, model, valid_findings)

    metrics = compute_metrics(ref_present, hyp_present)

    return {
        "study_id": study_id,
        "finding_f1": metrics["f1"],
        "finding_precision": metrics["precision"],
        "finding_recall": metrics["recall"],
        "ref_findings": sorted(ref_present),
        "hyp_findings": sorted(hyp_present),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "cost": c1 + c2,
    }


class FindingF1CTC:
    """Finding-level F1 extraction and evaluation for CT Chest."""

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        definitions_path: str | None = None,
        max_workers: int = 10,
    ):
        self.model = model
        self.max_workers = max_workers

        # Default to bundled definitions
        if definitions_path is None:
            definitions_path = os.path.join(
                os.path.dirname(__file__),
                "configs/definitions_lv010.jsonl",
            )

        # Load definitions
        defs = load_definitions(definitions_path)
        catalog = build_catalog(defs)

        # Build valid findings set for filtering
        self.valid_findings = {strip_version(d["finding_id"]) for d in defs}

        # Create CT chest-specific preamble
        preamble = (
            "# CT Chest — Finding Extraction\n\n"
            "You are an expert radiologist. For the CT chest report below,\n"
            "identify each finding that is mentioned and label it with a certainty\n"
            "value.\n\n"
            "Only classify findings mentioned in the body or impression of the report.\n"
            "Ignore findings in the clinical history or indication.\n\n"
            "## Certainty values\n\n"
            "- present      — the finding is asserted. Mild hedges (\"probable\", \"likely\",\n"
            "                 \"consistent with\", \"favors\") count as present.\n"
            "- absent       — the finding is explicitly negated, or explicitly stated as\n"
            "                 resolved.\n"
            "- uncertain    — significant hedging (\"possible\", \"could represent\",\n"
            "                 \"suggestive of\", \"cannot exclude\", \"non-specific for\"),\n"
            "                 ambiguous wording, or the finding appears in a differential\n"
            "                 with non-synonymous entities.\n"
            "- not changed  — the report asserts the finding is unchanged/stable from\n"
            "                 prior imaging and does not restate its current magnitude.\n\n"
            "\"not mentioned\" is never emitted — omit such findings entirely.\n"
        )

        # Override output format to enforce JSON
        json_output_instruction = (
            '\n\n## Output Format\n\n'
            'Return a JSON object with this exact structure:\n'
            '```json\n'
            '{\n'
            '  "findings": [\n'
            '    {"finding": "finding_id", "certainty": "present"},\n'
            '    {"finding": "finding_id", "certainty": "absent"},\n'
            '    ...\n'
            '  ]\n'
            '}\n'
            '```\n\n'
            'Do NOT use YAML or other formats. Return valid JSON only.'
        )

        self.system_prompt = f"{preamble}\n## Findings catalog\n\n{catalog}{json_output_instruction}"

    def __call__(
        self,
        refs: list[str],
        hyps: list[str],
        study_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract findings and compute F1 metrics for a batch of reports."""
        if study_ids is None:
            study_ids = [str(i) for i in range(len(refs))]

        per_sample = {}
        total_cost = 0.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    process_study,
                    ref,
                    hyp,
                    str(sid),
                    self.system_prompt,
                    self.model,
                    self.valid_findings,
                ): str(sid)
                for ref, hyp, sid in zip(refs, hyps, study_ids)
            }

            for future in as_completed(futures):
                result = future.result()
                sid = result.pop("study_id")
                cost = result.pop("cost")

                per_sample[sid] = result
                total_cost += cost

        # Compute aggregate stats
        all_f1 = [s["finding_f1"] for s in per_sample.values()]
        all_precision = [s["finding_precision"] for s in per_sample.values()]
        all_recall = [s["finding_recall"] for s in per_sample.values()]

        # Per-finding confusion matrix
        all_findings = set()
        for sample in per_sample.values():
            all_findings.update(sample["ref_findings"])
            all_findings.update(sample["hyp_findings"])

        per_finding_metrics = {}
        for finding in all_findings:
            tp = fp = fn = tn = 0
            for sample in per_sample.values():
                ref_set = set(sample["ref_findings"])
                hyp_set = set(sample["hyp_findings"])

                in_ref = finding in ref_set
                in_hyp = finding in hyp_set

                if in_ref and in_hyp:
                    tp += 1
                elif not in_ref and in_hyp:
                    fp += 1
                elif in_ref and not in_hyp:
                    fn += 1
                else:
                    tn += 1

            per_finding_metrics[finding] = compute_classification_metrics(tp, fp, fn, tn)

        # Macro-average across findings
        macro_f1 = sum(m["f1"] for m in per_finding_metrics.values()) / len(per_finding_metrics) if per_finding_metrics else 0.0
        macro_sens = sum(m["sensitivity"] for m in per_finding_metrics.values()) / len(per_finding_metrics) if per_finding_metrics else 0.0
        macro_spec = sum(m["specificity"] for m in per_finding_metrics.values()) / len(per_finding_metrics) if per_finding_metrics else 0.0
        macro_ppv = sum(m["ppv"] for m in per_finding_metrics.values()) / len(per_finding_metrics) if per_finding_metrics else 0.0
        macro_npv = sum(m["npv"] for m in per_finding_metrics.values()) / len(per_finding_metrics) if per_finding_metrics else 0.0

        # Micro-average (aggregate all TP/FP/FN/TN first)
        total_tp = sum(m["tp"] for m in per_finding_metrics.values())
        total_fp = sum(m["fp"] for m in per_finding_metrics.values())
        total_fn = sum(m["fn"] for m in per_finding_metrics.values())
        total_tn = sum(m["tn"] for m in per_finding_metrics.values())
        micro = compute_classification_metrics(total_tp, total_fp, total_fn, total_tn)

        return {
            "aggregate": {
                "finding_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
                "finding_precision": sum(all_precision) / len(all_precision) if all_precision else 0.0,
                "finding_recall": sum(all_recall) / len(all_recall) if all_recall else 0.0,
                "macro_f1": macro_f1,
                "macro_sensitivity": macro_sens,
                "macro_specificity": macro_spec,
                "macro_ppv": macro_ppv,
                "macro_npv": macro_npv,
                "micro_f1": micro["f1"],
                "micro_sensitivity": micro["sensitivity"],
                "micro_specificity": micro["specificity"],
                "micro_ppv": micro["ppv"],
                "micro_npv": micro["npv"],
                "micro_tp": micro["tp"],
                "micro_fp": micro["fp"],
                "micro_fn": micro["fn"],
                "micro_tn": micro["tn"],
                "total_cost_usd": total_cost,
                "n_findings": len(all_findings),
            },
            "per_sample": per_sample,
            "per_finding": per_finding_metrics,
        }
