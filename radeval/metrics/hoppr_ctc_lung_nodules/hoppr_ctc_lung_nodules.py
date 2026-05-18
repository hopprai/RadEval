"""HopprCTCLungNodules scorer — LLM-as-judge metric for the PULMONARY NODULES
subsection of CT findings reports.

Takes full clean_findings strings (ref, hyp), extracts the PULMONARY NODULES
segment from each, sends them to a judge LLM, parses the structured JSON
response, and computes per-row metrics deterministically.

Scope (VLM / text-generation setting):
    - Detection (per size bucket lt4, 4to6, 6to8, 8to15, gt15) with
      sensitivity and FP-per-study aggregated at corpus level.
    - Density two-class accuracy (solid vs sub-solid).
    - Calcified accuracy.
    - Diameter MAE (absolute, per bucket) and MAPE (relative).
    - Laterality and lobe accuracy.
    - Slice-number criterion is intentionally DROPPED — our reports only
      contain lobe/laterality/segment language, not slice indices.
    - Volume scoring is DEFERRED — the schema does not currently emit
      volumes; the head-of-ML rebuttal notes that volume error scales
      ~3x diameter error so a future extension can derive a volume KPI
      from the existing diameter MAE buckets.

Supported providers: openai, gemini.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from .._llm_base import LLMMetricBase
from .prompt_parts import SYSTEM_MSG, build_prompt
from .utils import (
    SIZE_BUCKETS,
    canonicalize_pn_segment,
    compute_per_row_metrics,
    empty_row_result,
    extract_pn_segment,
    parse_json_response,
    validate_response,
)

logger = logging.getLogger(__name__)


# Per-row rate metrics that are valid floats (skip None when aggregating).
_METRIC_KEYS: tuple[str, ...] = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "size_accuracy",
    "size_exact_match",
    "size_mae_mm",
    "size_mape",
    "density_accuracy_solid",
    "density_accuracy_subsolid",
    "density_accuracy_micro",
    "calcified_accuracy",
    "laterality_accuracy",
    "location_accuracy",
    "noun_accuracy",
    "uncertainty_accuracy",
    "type_accuracy",
    "composite",
)

# Per-row counters that the adapter aggregates as corpus-level sums.
_COUNT_KEYS: tuple[str, ...] = (
    "n_reference",
    "n_predicted",
    "n_matched",
    "n_false_findings",
    "n_missing_findings",
    "n_size_pairs",
    "n_size_errors",
    "n_density_errors",
    "n_calcified_errors",
    "n_location_errors",
    "n_noun_errors",
    "n_uncertainty_errors",
    "density_correct_solid",
    "density_total_solid",
    "density_correct_subsolid",
    "density_total_subsolid",
    "calcified_correct",
    "calcified_total",
    "laterality_correct",
    "laterality_total",
    "lobe_correct",
    "lobe_total",
)

# Per-bucket counters (all integer-valued; floats for the abs/pct sums).
_BUCKET_KEYS: tuple[str, ...] = tuple(
    f"{prefix}_{b}"
    for b in SIZE_BUCKETS
    for prefix in ("tp", "fn", "fp", "abs_err_sum", "abs_err_n",
                   "pct_err_sum", "pct_err_n")
)


def _mean_skip_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_skip_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(np.std(clean))


class HopprCTCLungNodulesScore(LLMMetricBase):
    """LLM-as-judge scorer for the PULMONARY NODULES subsection.

    Usage:
        scorer = HopprCTCLungNodulesScore(provider="gemini")
        mean_composite, std_composite, per_row_scores, results_df = scorer(refs, hyps)

    `per_row_scores` is a list of composite scores (one per input pair).
    `results_df` is a DataFrame with one row per sample carrying every
    per-row rate metric, raw error count, and per-bucket detection /
    diameter-error counter exposed by `compute_per_row_metrics`.

    Default provider is gemini; gpt-4o-mini is supported as a fallback.
    """

    SUPPORTED_PROVIDERS: ClassVar[set[str]] = {"openai", "gemini"}

    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        max_concurrent: int = 50,
    ):
        if model_name is None:
            model_name = (
                self.DEFAULT_OPENAI_MODEL if provider == "openai"
                else self.DEFAULT_GEMINI_MODEL
            )
        super().__init__(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )

    def _build_request(self, ref: str, hyp: str, **kwargs) -> dict[str, Any]:
        """Build the provider-specific request payload.

        ref and hyp are full clean_findings strings; we extract the
        PULMONARY NODULES segment and canonicalize sentence order before
        prompting so the judge sees the same input regardless of how the
        source text ordered the nodules.
        """
        ref_pn = canonicalize_pn_segment(extract_pn_segment(ref))
        hyp_pn = canonicalize_pn_segment(extract_pn_segment(hyp))
        prompt = build_prompt(ref_pn, hyp_pn)

        if self.provider == "openai":
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "seed": 42,
                "response_format": {"type": "json_object"},
            }
        elif self.provider == "gemini":
            from google.genai import types
            return {
                "contents": prompt,
                "config": types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_MSG,
                ),
            }
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    def _parse_response(self, raw: str) -> dict:
        data = parse_json_response(raw)
        validate_response(data)
        return compute_per_row_metrics(data)

    def _short_circuit(self, ref: str, hyp: str) -> Optional[dict]:
        """Return a pre-computed result if the row doesn't need an LLM call."""
        ref_pn = extract_pn_segment(ref)
        hyp_pn = extract_pn_segment(hyp)
        if not ref_pn and not hyp_pn:
            return empty_row_result(both_empty=True)
        return None

    def _evaluate_one(self, ref, hyp, max_retries=2, **kwargs):
        sc = self._short_circuit(ref, hyp)
        if sc is not None:
            return sc
        try:
            return super()._evaluate_one(ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError as e:
            logger.error("hoppr_ctc_lung_nodules: all attempts failed for one sample: %s", e)
            return self._nan_fallback()

    async def _evaluate_one_async(self, ref, hyp, max_retries=2, **kwargs):
        sc = self._short_circuit(ref, hyp)
        if sc is not None:
            return sc
        try:
            return await super()._evaluate_one_async(ref, hyp, max_retries=max_retries, **kwargs)
        except RuntimeError as e:
            logger.error("hoppr_ctc_lung_nodules: all async attempts failed: %s", e)
            return self._nan_fallback()

    @staticmethod
    def _nan_fallback() -> dict:
        """Per-row result to use when the LLM call fails persistently."""
        result = {k: None for k in _METRIC_KEYS}
        for k in _COUNT_KEYS:
            result[k] = 0
        for k in _BUCKET_KEYS:
            result[k] = 0 if k.endswith("_n") or k.startswith(("tp_", "fn_", "fp_")) else 0.0
        return result

    def _aggregate(self, results, refs, hyps):
        """Aggregate per-row dicts into (mean_composite, std_composite,
        per_sample_composite, results_df)."""
        composite_per_row = [r.get("composite") for r in results]
        mean_composite = _mean_skip_none(composite_per_row) or 0.0
        std_composite = _std_skip_none(composite_per_row) or 0.0

        rows = []
        for ref, hyp, r in zip(refs, hyps, results):
            row = {"reference": ref, "prediction": hyp}
            for k in _METRIC_KEYS:
                row[k] = r.get(k)
            for k in _COUNT_KEYS:
                row[k] = r.get(k, 0)
            for k in _BUCKET_KEYS:
                default = 0.0 if k.startswith(("abs_err_sum_", "pct_err_sum_")) else 0
                row[k] = r.get(k, default)
            rows.append(row)
        results_df = pd.DataFrame(rows)

        return mean_composite, std_composite, composite_per_row, results_df


class HopprCTCLungNodules(HopprCTCLungNodulesScore):
    """Alias of HopprCTCLungNodulesScore."""
