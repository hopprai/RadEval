"""RadEval-facing adapter for the hoppr_ctc_lung_nodules metric.

Wraps HopprCTCLungNodulesScore and emits per-dimension output keys in the
shape expected by RadEval.compute_scores() across all three output modes
(default, per_sample, detailed).

Aggregation strategy:
    - Rate-style metrics that the scorer already emits per row
      (`detection_*`, `size_accuracy`, `composite`, etc.) are aggregated as
      means of per-row rates, matching legacy behavior.
    - Desiderata-aligned KPIs (per-bucket sensitivity / FP-per-study,
      per-bucket diameter MAE/MAPE, density two-class accuracy, calcified,
      laterality, lobe) are aggregated as **corpus-level rates**: sum of
      hits divided by sum of denominators across all rows. This matches
      how the desiderata are framed (e.g. "FP per study" is total FPs
      divided by number of studies, not the mean of per-study FP rates).
    - Per-bucket diameter MAE / MAPE are corpus-level means of absolute
      errors (sum of errors / count of pairs) within each ref bucket.

Detailed mode also emits per-KPI pass/fail booleans against the
consolidated lung-nodule desiderata and an overall
`hoppr_ctc_lung_nodules_desiderata_pass` aggregate.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .._base import MetricBase
from .utils import SIZE_BUCKETS

logger = logging.getLogger(__name__)


# Legacy / per-row rate keys carried over from the prior nodule_eval adapter.
# These remain `mean of per-row rates` for backward compatibility.
_LEGACY_RATE_KEYS: dict[str, str] = {
    "hoppr_ctc_lung_nodules_detection_precision": "detection_precision",
    "hoppr_ctc_lung_nodules_detection_recall":    "detection_recall",
    "hoppr_ctc_lung_nodules_detection_f1":        "detection_f1",
    "hoppr_ctc_lung_nodules_size_accuracy":       "size_accuracy",
    "hoppr_ctc_lung_nodules_size_exact_match":    "size_exact_match",
    "hoppr_ctc_lung_nodules_size_mae_mm":         "size_mae_mm",
    "hoppr_ctc_lung_nodules_size_mape":           "size_mape",
    "hoppr_ctc_lung_nodules_noun_accuracy":       "noun_accuracy",
    "hoppr_ctc_lung_nodules_uncertainty_accuracy": "uncertainty_accuracy",
    "hoppr_ctc_lung_nodules_composite":           "composite",
}


# Desiderata thresholds (see module docstring of hoppr_ctc_lung_nodules.py).
# Values are the targets the metric must meet/clear to pass.
_DESIDERATA: dict[str, dict[str, Any]] = {
    "sensitivity_4to6":  {"target": 0.60, "direction": "ge"},
    "sensitivity_6to8":  {"target": 0.85, "direction": "ge"},
    "sensitivity_8to15": {"target": 0.95, "direction": "ge"},
    "sensitivity_gt15":  {"target": 0.95, "direction": "ge"},
    "fp_per_study_4to6":  {"target": 2.00, "direction": "le"},
    "fp_per_study_6to8":  {"target": 2.00, "direction": "le"},
    "fp_per_study_8to15": {"target": 0.75, "direction": "le"},
    "fp_per_study_gt15":  {"target": 0.75, "direction": "le"},
    "diameter_mae_4to6":  {"target": 1.00, "direction": "le"},
    "diameter_mae_6to8":  {"target": 1.25, "direction": "le"},
    "diameter_mape_gt8":  {"target": 0.15, "direction": "le"},
    "density_acc_solid":    {"target": 0.70, "direction": "ge"},
    "density_acc_subsolid": {"target": 0.70, "direction": "ge"},
}


def _mean_skip_none(values) -> float | None:
    clean = [v for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_skip_none(values) -> float | None:
    clean = [v for v in values
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(np.std(clean))


def _safe_div(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def _passes(value, target, direction) -> bool | None:
    if value is None:
        return None
    if direction == "ge":
        return float(value) >= float(target)
    if direction == "le":
        return float(value) <= float(target)
    return None


def _emit(out: dict, key: str, rate: float | None, support: int | float) -> None:
    """Write a rate-style KPI and its `_support` denominator to `out`.

    `support` is the count of underlying samples that contributed to the
    rate (e.g. for sensitivity_4to6 it is TP + FN within the 4to6 bucket;
    for diameter_mae_4to6 it is the number of TP pairs with both numeric
    sizes in the 4to6 bucket; for fp_per_study_<bucket> it is n_studies).
    Always emits both the rate and the support so downstream consumers
    can sanity-check rates with tiny denominators.
    """
    out[key] = round(rate, 4) if rate is not None else None
    out[f"{key}_support"] = int(support)


def _support_count(values) -> int:
    """Count non-None / non-NaN entries — used as the support for legacy
    rate keys aggregated as means of per-row rates."""
    return sum(
        1
        for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    )


class HopprCTCLungNodulesMetric(MetricBase):
    name = "hoppr_ctc_lung_nodules"
    display_name = "HopprCTCLungNodules"
    is_api_based = True

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        max_concurrent: int = 50,
        cache_dir: str | None = None,
        **kwargs,
    ):
        from . import HopprCTCLungNodulesScore
        if HopprCTCLungNodulesScore is None:
            raise ImportError(
                "HopprCTCLungNodules failed to import — missing dependency. "
                "See radeval/metrics/hoppr_ctc_lung_nodules/__init__.py.")
        self._scorer = HopprCTCLungNodulesScore(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            max_concurrent=max_concurrent,
        )

    @property
    def cost_tracker(self):
        return getattr(self._scorer, "cost_tracker", None)

    def metric_keys(self, detailed: bool = False) -> list[str]:
        keys: list[str] = []
        for k in self._all_aggregate_keys():
            keys.append(k)
            # n_studies has no companion `_support` (it IS the support count
            # for the whole study population).
            if k != "hoppr_ctc_lung_nodules_n_studies":
                keys.append(f"{k}_support")
        if detailed:
            keys.extend([k + "_std" for k in _LEGACY_RATE_KEYS])
            keys.extend([f"{k}_pass" for k in (
                f"hoppr_ctc_lung_nodules_{name}" for name in _DESIDERATA
            )])
            keys.append("hoppr_ctc_lung_nodules_desiderata_pass")
        return keys

    @staticmethod
    def _all_aggregate_keys() -> list[str]:
        keys = list(_LEGACY_RATE_KEYS.keys())
        for b in SIZE_BUCKETS:
            keys.append(f"hoppr_ctc_lung_nodules_sensitivity_{b}")
            keys.append(f"hoppr_ctc_lung_nodules_fp_per_study_{b}")
        for b in ("4to6", "6to8"):
            keys.append(f"hoppr_ctc_lung_nodules_diameter_mae_{b}")
        keys.append("hoppr_ctc_lung_nodules_diameter_mape_gt8")
        keys.extend([
            "hoppr_ctc_lung_nodules_density_acc_solid",
            "hoppr_ctc_lung_nodules_density_acc_subsolid",
            "hoppr_ctc_lung_nodules_density_acc_macro",
            "hoppr_ctc_lung_nodules_calcified_accuracy",
            "hoppr_ctc_lung_nodules_laterality_accuracy",
            "hoppr_ctc_lung_nodules_lobe_accuracy",
            "hoppr_ctc_lung_nodules_n_studies",
        ])
        return keys

    def compute(
        self,
        refs: list[str],
        hyps: list[str],
        per_sample: bool = False,
        detailed: bool = False,
        on_progress=None,
    ) -> dict[str, Any]:
        """Run scoring and return output in the requested mode."""
        _, _, _, results_df = self._scorer(
            refs, hyps, on_sample_done=on_progress,
        )

        if per_sample:
            return self._per_sample_output(results_df)

        return self._aggregate_output(results_df, detailed=detailed)

    @staticmethod
    def _per_sample_output(df: pd.DataFrame) -> dict[str, Any]:
        """Per-sample mode: return one list per legacy rate key, plus a
        matching `_support` list of 0/1 indicators (1 = row contributed
        to this KPI's denominator, 0 = row's per-row rate was undefined
        and therefore skipped in the corpus aggregate).

        Per-bucket KPIs are not meaningful per-sample (e.g. a study with
        zero ref nodules in a bucket has undefined sensitivity), so they
        are omitted in this mode.
        """
        out: dict[str, Any] = {}
        for out_key, col in _LEGACY_RATE_KEYS.items():
            series = df[col].tolist() if col in df.columns else []
            out[out_key] = series
            out[f"{out_key}_support"] = [
                int(v is not None and not (isinstance(v, float) and np.isnan(v)))
                for v in series
            ]
        return out

    def _aggregate_output(self, df: pd.DataFrame, detailed: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # Legacy rate keys: mean of per-row rates. Support = number of
        # rows where the per-row rate was defined (non-None / non-NaN).
        for out_key, col in _LEGACY_RATE_KEYS.items():
            series = df[col].tolist() if col in df.columns else []
            mean = _mean_skip_none(series)
            support = _support_count(series)
            _emit(out, out_key, mean, support)
            if detailed:
                std = _std_skip_none(series)
                out[out_key + "_std"] = round(std, 4) if std is not None else None

        n_studies = len(df)
        out["hoppr_ctc_lung_nodules_n_studies"] = n_studies

        # Per-bucket sensitivity (support = TP + FN in bucket) and
        # FP-per-study (support = n_studies; same for every bucket).
        for b in SIZE_BUCKETS:
            tp = float(df.get(f"tp_{b}", pd.Series(dtype=float)).sum() or 0)
            fn = float(df.get(f"fn_{b}", pd.Series(dtype=float)).sum() or 0)
            fp = float(df.get(f"fp_{b}", pd.Series(dtype=float)).sum() or 0)
            sens = _safe_div(tp, tp + fn)
            fp_per_study = _safe_div(fp, n_studies)
            _emit(
                out,
                f"hoppr_ctc_lung_nodules_sensitivity_{b}",
                sens,
                support=int(tp + fn),
            )
            _emit(
                out,
                f"hoppr_ctc_lung_nodules_fp_per_study_{b}",
                fp_per_study,
                support=n_studies,
            )

        # Per-bucket diameter MAE (4to6, 6to8). Support = number of TP
        # pairs in the bucket where BOTH sides had a numeric size.
        for b in ("4to6", "6to8"):
            num = float(df.get(f"abs_err_sum_{b}", pd.Series(dtype=float)).sum() or 0)
            den = float(df.get(f"abs_err_n_{b}", pd.Series(dtype=float)).sum() or 0)
            mae = _safe_div(num, den)
            _emit(out, f"hoppr_ctc_lung_nodules_diameter_mae_{b}", mae, support=int(den))

        # >8 mm MAPE pools 8to15 and gt15. Support = pooled TP-pairs count.
        gt8_num = float(
            (df.get("pct_err_sum_8to15", pd.Series(dtype=float)).sum() or 0)
            + (df.get("pct_err_sum_gt15", pd.Series(dtype=float)).sum() or 0)
        )
        gt8_den = float(
            (df.get("pct_err_n_8to15", pd.Series(dtype=float)).sum() or 0)
            + (df.get("pct_err_n_gt15", pd.Series(dtype=float)).sum() or 0)
        )
        mape_gt8 = _safe_div(gt8_num, gt8_den)
        _emit(
            out,
            "hoppr_ctc_lung_nodules_diameter_mape_gt8",
            mape_gt8,
            support=int(gt8_den),
        )

        # Density two-class accuracy. Support = number of TP pairs where
        # BOTH sides had a non-null density AND ref resolves to that class.
        d_corr_s = float(df.get("density_correct_solid", pd.Series(dtype=float)).sum() or 0)
        d_tot_s = float(df.get("density_total_solid", pd.Series(dtype=float)).sum() or 0)
        d_corr_ns = float(df.get("density_correct_subsolid", pd.Series(dtype=float)).sum() or 0)
        d_tot_ns = float(df.get("density_total_subsolid", pd.Series(dtype=float)).sum() or 0)
        acc_solid = _safe_div(d_corr_s, d_tot_s)
        acc_subsolid = _safe_div(d_corr_ns, d_tot_ns)
        _emit(out, "hoppr_ctc_lung_nodules_density_acc_solid",
              acc_solid, support=int(d_tot_s))
        _emit(out, "hoppr_ctc_lung_nodules_density_acc_subsolid",
              acc_subsolid, support=int(d_tot_ns))

        if acc_solid is not None and acc_subsolid is not None:
            macro = (acc_solid + acc_subsolid) / 2.0
        elif acc_solid is not None:
            macro = acc_solid
        elif acc_subsolid is not None:
            macro = acc_subsolid
        else:
            macro = None
        _emit(out, "hoppr_ctc_lung_nodules_density_acc_macro",
              macro, support=int(d_tot_s + d_tot_ns))

        # Calcified, laterality, lobe accuracy. Support = TP pairs with
        # both sides having a non-null annotation for that attribute.
        for out_key, num_col, den_col in [
            ("hoppr_ctc_lung_nodules_calcified_accuracy",
             "calcified_correct", "calcified_total"),
            ("hoppr_ctc_lung_nodules_laterality_accuracy",
             "laterality_correct", "laterality_total"),
            ("hoppr_ctc_lung_nodules_lobe_accuracy",
             "lobe_correct", "lobe_total"),
        ]:
            num = float(df.get(num_col, pd.Series(dtype=float)).sum() or 0)
            den = float(df.get(den_col, pd.Series(dtype=float)).sum() or 0)
            acc = _safe_div(num, den)
            _emit(out, out_key, acc, support=int(den))

        if detailed:
            self._add_desiderata_flags(out)

        return out

    @staticmethod
    def _add_desiderata_flags(out: dict[str, Any]) -> None:
        """Mutate `out` to include per-KPI pass booleans + overall pass."""
        all_pass = True
        any_known = False
        for name, spec in _DESIDERATA.items():
            kpi_key = f"hoppr_ctc_lung_nodules_{name}"
            value = out.get(kpi_key)
            passed = _passes(value, spec["target"], spec["direction"])
            out[f"{kpi_key}_pass"] = passed
            if passed is None:
                continue
            any_known = True
            if not passed:
                all_pass = False
        out["hoppr_ctc_lung_nodules_desiderata_pass"] = (
            all_pass if any_known else None
        )

