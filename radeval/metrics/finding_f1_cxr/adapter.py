"""RadEval adapter for Finding F1 CXR metric."""
from radeval.metrics._base import MetricBase
from .finding_f1_cxr import FindingF1CXR


class FindingF1CXRMetric(MetricBase):
    """Finding-level F1 extraction and evaluation for CXR using lv021 Phase 1.

    Includes clinically-weighted metrics where critical findings (e.g. pneumothorax,
    aortic dissection) contribute more to the score than incidental findings.
    """

    name = "finding_f1_cxr"
    display_name = "Finding F1 (CXR lv021)"

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        max_workers: int = 10,
    ):
        self._scorer = FindingF1CXR(model=model, max_workers=max_workers)

    def metric_keys(self, detailed: bool = False) -> list[str]:
        """Return output keys this metric produces."""
        keys = [
            "finding_f1",
            "finding_precision",
            "finding_recall",
            "weighted_finding_f1",
            "weighted_finding_precision",
            "weighted_finding_recall",
            "macro_f1",
            "macro_sensitivity",
            "macro_specificity",
            "macro_ppv",
            "macro_npv",
            "weighted_macro_f1",
            "weighted_macro_sensitivity",
            "weighted_macro_specificity",
            "weighted_macro_ppv",
            "weighted_macro_npv",
            "micro_f1",
            "micro_sensitivity",
            "micro_specificity",
            "micro_ppv",
            "micro_npv",
            "weighted_micro_f1",
            "weighted_micro_sensitivity",
            "weighted_micro_specificity",
            "weighted_micro_ppv",
            "weighted_micro_npv",
        ]
        if detailed:
            keys.extend([
                "micro_tp",
                "micro_fp",
                "micro_fn",
                "micro_tn",
                "n_findings",
                "total_cost_usd",
            ])
        return keys

    def _compute_raw(self, refs, hyps, on_progress=None):
        """Compute finding F1 metrics."""
        result = self._scorer(refs=refs, hyps=hyps)

        # Extract per-sample scores
        per_sample_scores = {}
        for key in ["finding_f1", "finding_precision", "finding_recall",
                    "weighted_f1", "weighted_precision", "weighted_recall"]:
            per_sample_scores[key] = [
                s[key] for s in result["per_sample"].values()
            ]

        agg = result["aggregate"]

        return {
            "finding_f1": {
                "aggregate": agg["finding_f1"],
                "per_sample": per_sample_scores["finding_f1"],
            },
            "finding_precision": {
                "aggregate": agg["finding_precision"],
                "per_sample": per_sample_scores["finding_precision"],
            },
            "finding_recall": {
                "aggregate": agg["finding_recall"],
                "per_sample": per_sample_scores["finding_recall"],
            },
            "weighted_finding_f1": {
                "aggregate": agg["weighted_finding_f1"],
                "per_sample": per_sample_scores["weighted_f1"],
            },
            "weighted_finding_precision": {
                "aggregate": agg["weighted_finding_precision"],
                "per_sample": per_sample_scores["weighted_precision"],
            },
            "weighted_finding_recall": {
                "aggregate": agg["weighted_finding_recall"],
                "per_sample": per_sample_scores["weighted_recall"],
            },
            "macro_f1": {"aggregate": agg["macro_f1"]},
            "macro_sensitivity": {"aggregate": agg["macro_sensitivity"]},
            "macro_specificity": {"aggregate": agg["macro_specificity"]},
            "macro_ppv": {"aggregate": agg["macro_ppv"]},
            "macro_npv": {"aggregate": agg["macro_npv"]},
            "weighted_macro_f1": {"aggregate": agg["weighted_macro_f1"]},
            "weighted_macro_sensitivity": {"aggregate": agg["weighted_macro_sensitivity"]},
            "weighted_macro_specificity": {"aggregate": agg["weighted_macro_specificity"]},
            "weighted_macro_ppv": {"aggregate": agg["weighted_macro_ppv"]},
            "weighted_macro_npv": {"aggregate": agg["weighted_macro_npv"]},
            "micro_f1": {"aggregate": agg["micro_f1"]},
            "micro_sensitivity": {"aggregate": agg["micro_sensitivity"]},
            "micro_specificity": {"aggregate": agg["micro_specificity"]},
            "micro_ppv": {"aggregate": agg["micro_ppv"]},
            "micro_npv": {"aggregate": agg["micro_npv"]},
            "weighted_micro_f1": {"aggregate": agg["weighted_micro_f1"]},
            "weighted_micro_sensitivity": {"aggregate": agg["weighted_micro_sensitivity"]},
            "weighted_micro_specificity": {"aggregate": agg["weighted_micro_specificity"]},
            "weighted_micro_ppv": {"aggregate": agg["weighted_micro_ppv"]},
            "weighted_micro_npv": {"aggregate": agg["weighted_micro_npv"]},
            "micro_tp": {"aggregate": agg["micro_tp"]},
            "micro_fp": {"aggregate": agg["micro_fp"]},
            "micro_fn": {"aggregate": agg["micro_fn"]},
            "micro_tn": {"aggregate": agg["micro_tn"]},
            "n_findings": {"aggregate": agg["n_findings"]},
            "total_cost_usd": {"aggregate": agg["total_cost_usd"]},
        }
