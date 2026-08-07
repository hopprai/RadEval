"""RadEval adapter for Finding F1 CXR metric."""
from radeval.metrics._base import MetricBase
from .finding_f1_cxr import FindingF1CXR


class FindingF1CXRMetric(MetricBase):
    """Finding-level F1 extraction and evaluation for CXR using lv021 Phase 1.

    Returns per-sample F1/precision/recall, aggregate metrics, and per-finding confusion matrices.
    Outputs: finding_f1, finding_precision, finding_recall, macro/micro sensitivity/specificity/PPV/NPV.
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
            "macro_f1",
            "macro_sensitivity",
            "macro_specificity",
            "macro_ppv",
            "macro_npv",
            "micro_f1",
            "micro_sensitivity",
            "micro_specificity",
            "micro_ppv",
            "micro_npv",
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
        for key in ["finding_f1", "finding_precision", "finding_recall"]:
            per_sample_scores[key] = [
                s[key] for s in result["per_sample"].values()
            ]

        return {
            "finding_f1": {
                "aggregate": result["aggregate"]["finding_f1"],
                "per_sample": per_sample_scores["finding_f1"],
            },
            "finding_precision": {
                "aggregate": result["aggregate"]["finding_precision"],
                "per_sample": per_sample_scores["finding_precision"],
            },
            "finding_recall": {
                "aggregate": result["aggregate"]["finding_recall"],
                "per_sample": per_sample_scores["finding_recall"],
            },
            "macro_f1": {
                "aggregate": result["aggregate"]["macro_f1"],
            },
            "macro_sensitivity": {
                "aggregate": result["aggregate"]["macro_sensitivity"],
            },
            "macro_specificity": {
                "aggregate": result["aggregate"]["macro_specificity"],
            },
            "macro_ppv": {
                "aggregate": result["aggregate"]["macro_ppv"],
            },
            "macro_npv": {
                "aggregate": result["aggregate"]["macro_npv"],
            },
            "micro_f1": {
                "aggregate": result["aggregate"]["micro_f1"],
            },
            "micro_sensitivity": {
                "aggregate": result["aggregate"]["micro_sensitivity"],
            },
            "micro_specificity": {
                "aggregate": result["aggregate"]["micro_specificity"],
            },
            "micro_ppv": {
                "aggregate": result["aggregate"]["micro_ppv"],
            },
            "micro_npv": {
                "aggregate": result["aggregate"]["micro_npv"],
            },
            "micro_tp": {
                "aggregate": result["aggregate"]["micro_tp"],
            },
            "micro_fp": {
                "aggregate": result["aggregate"]["micro_fp"],
            },
            "micro_fn": {
                "aggregate": result["aggregate"]["micro_fn"],
            },
            "micro_tn": {
                "aggregate": result["aggregate"]["micro_tn"],
            },
            "n_findings": {
                "aggregate": result["aggregate"]["n_findings"],
            },
            "total_cost_usd": {
                "aggregate": result["aggregate"]["total_cost_usd"],
            },
        }
