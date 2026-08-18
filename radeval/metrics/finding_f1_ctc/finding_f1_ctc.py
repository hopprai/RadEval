"""Finding-level F1 extraction and evaluation for CT Chest."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .llm_client import call_llm

# 23 findings + 2 regions, matching ctc_config.yaml / CTChestFindingTree
C1_FINDINGS = [
    "acute_aortic_injury",
    "acute_pulmonary_embolism",
    "acute_rib_fracture",
    "air_space_opacity",
    "aortic_aneurysm",
    "bone_lesion",
    "chronic_pulmonary_embolism",
    "hilar_lymphadenopathy",
    "lung_nodule_or_mass",
    "mediastinal_lymphadenopathy",
    "pericardial_effusion",
    "pleural_effusion",
    "pneumothorax",
]

C2_FINDINGS = [
    "atelectasis",
    "bronchiectasis",
    "cardiomegaly",
    "copd_emphysema",
    "interlobular_septal_thickening",
    "vertebral_compression_fracture",
]

C3_FINDINGS = [
    "aortic_atherosclerosis",
    "aortic_valve_calcification",
    "coronary_artery_calcifications",
    "prior_myocardial_infarction",
]

REGION_FINDINGS = [
    "neck_base",
    "upper_abdomen",
]

ALL_FINDINGS = C1_FINDINGS + C2_FINDINGS + C3_FINDINGS + REGION_FINDINGS

SYSTEM_PROMPT = """\
You are an expert radiologist performing structured extraction of findings from CT chest reports.
Classify each finding strictly from the report body and impression.
Never use clinical history or indication.

## Label Categories

**C1 — Consistently Labeled** (absent if not mentioned):
`acute_aortic_injury`, `acute_pulmonary_embolism`, `acute_rib_fracture`, `air_space_opacity`,
`aortic_aneurysm`, `bone_lesion`, `chronic_pulmonary_embolism`, `hilar_lymphadenopathy`,
`lung_nodule_or_mass`, `mediastinal_lymphadenopathy`, `pericardial_effusion`, `pleural_effusion`, `pneumothorax`
- present / absent (explicitly negated OR not mentioned) / uncertain (hedged language)

**C2 — Anatomy-Gated** (absent if relevant anatomy is normal; "not reported" if anatomy not addressed):
`atelectasis`, `bronchiectasis`, `cardiomegaly`, `copd_emphysema`,
`interlobular_septal_thickening`, `vertebral_compression_fracture`
- present / absent (explicitly negated OR relevant anatomy described as normal/unremarkable/clear) / uncertain / "not changed" / "not reported"

**C3 — Text-Only** (not reported if not mentioned; absent only if explicitly negated):
`aortic_atherosclerosis`, `aortic_valve_calcification`, `coronary_artery_calcifications`,
`prior_myocardial_infarction`
- present / absent (explicitly negated only — never infer from normal anatomy) / uncertain / "not changed" / "not reported"

**Region fields** (`neck_base`, `upper_abdomen`) use a separate label set:
- normal (region addressed and described as normal/unremarkable)
- abnormal (any abnormality described in the region)
- "not reported" (region not mentioned)

Use "not changed" (C2/C3) when the report's only qualifier is unchanged/stable/no interval change from prior, with no new positive features. Prefer "not changed" over "present" in that case.

## C2 — absent vs "not reported"
1. Is the relevant anatomy mentioned in the report?
   - No → **"not reported"**
   - Yes → continue
2. Is it described as normal, unremarkable, or clear?
   - Yes → **absent**
   - No → assign present / uncertain based on what is stated

Never assign **absent** solely because a finding is unmentioned — only when the anatomy is addressed and described favourably.

### C2 Anatomy Gate
| Finding | Relevant anatomy — gate phrases |
|---|---|
| `atelectasis`, `bronchiectasis`, `copd_emphysema`, `interlobular_septal_thickening` | lungs, pulmonary parenchyma, airways — "lungs clear/unremarkable", "no acute cardiopulmonary process" |
| `cardiomegaly` | heart, cardiac silhouette — "heart normal in size", "cardiac silhouette unremarkable" |
| `vertebral_compression_fracture` | spine, vertebral bodies, osseous structures — "spine unremarkable", "no acute osseous abnormality" |

## Output Format

Return a JSON object with exactly one key per finding. For C1/C2/C3 findings, the value is the presence label. For region fields, the value is the region label.

```json
{
  "acute_aortic_injury": "absent",
  "acute_pulmonary_embolism": "absent",
  "acute_rib_fracture": "absent",
  "air_space_opacity": "present",
  "aortic_aneurysm": "absent",
  "bone_lesion": "absent",
  "chronic_pulmonary_embolism": "absent",
  "hilar_lymphadenopathy": "absent",
  "lung_nodule_or_mass": "present",
  "mediastinal_lymphadenopathy": "absent",
  "pericardial_effusion": "absent",
  "pleural_effusion": "absent",
  "pneumothorax": "absent",
  "atelectasis": "absent",
  "bronchiectasis": "not reported",
  "cardiomegaly": "absent",
  "copd_emphysema": "absent",
  "interlobular_septal_thickening": "not reported",
  "vertebral_compression_fracture": "absent",
  "aortic_atherosclerosis": "not reported",
  "aortic_valve_calcification": "not reported",
  "coronary_artery_calcifications": "not reported",
  "prior_myocardial_infarction": "not reported",
  "neck_base": "not reported",
  "upper_abdomen": "abnormal"
}
```

Return valid JSON only. Every finding must be present in the output."""


def parse_structured_output(raw: str) -> dict[str, str]:
    """Parse the structured JSON output from LLM into finding→presence mapping."""
    raw = raw.strip()
    if not raw:
        return {}

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:(-1 if lines[-1].strip() == "```" else len(lines))])
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {k: v for k, v in parsed.items() if isinstance(v, str)}
    except json.JSONDecodeError:
        pass

    return {}


def extract_positive_findings(
    report: str,
    model: str,
) -> tuple[set[str], float]:
    """Extract findings classified as 'present' (or 'abnormal' for regions) from a report."""
    try:
        raw, cost = call_llm(
            prompt=f"## Report\n\n{report}\n",
            system_prompt=SYSTEM_PROMPT,
            model_name=model,
            temperature=0.0,
            max_tokens=4000,
        )
    except Exception as e:
        print(f"ERROR calling LLM: {e}")
        return set(), 0.0

    labels = parse_structured_output(raw)

    present = set()
    for finding_id, label in labels.items():
        if finding_id not in set(ALL_FINDINGS):
            continue
        if finding_id in REGION_FINDINGS:
            if label == "abnormal":
                present.add(finding_id)
        else:
            if label == "present":
                present.add(finding_id)

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
    model: str,
) -> dict[str, Any]:
    """Process a single study — extract findings from ref and hyp, compute metrics."""
    ref_present, c1 = extract_positive_findings(ref, model)
    hyp_present, c2 = extract_positive_findings(hyp, model)

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
        max_workers: int = 10,
    ):
        self.model = model
        self.max_workers = max_workers

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
                    self.model,
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

        # Per-finding confusion matrix across all 25 findings
        valid_findings = set(ALL_FINDINGS)
        per_finding_metrics = {}
        for finding in valid_findings:
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
        n_findings = len(per_finding_metrics)
        macro_f1 = sum(m["f1"] for m in per_finding_metrics.values()) / n_findings if n_findings else 0.0
        macro_sens = sum(m["sensitivity"] for m in per_finding_metrics.values()) / n_findings if n_findings else 0.0
        macro_spec = sum(m["specificity"] for m in per_finding_metrics.values()) / n_findings if n_findings else 0.0
        macro_ppv = sum(m["ppv"] for m in per_finding_metrics.values()) / n_findings if n_findings else 0.0
        macro_npv = sum(m["npv"] for m in per_finding_metrics.values()) / n_findings if n_findings else 0.0

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
                "n_findings": n_findings,
            },
            "per_sample": per_sample,
            "per_finding": per_finding_metrics,
        }
