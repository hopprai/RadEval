from .._base import MetricBase


class F1HopprCthRrgLv001ClsLv003Metric(MetricBase):
    name = "f1hoppr_cth_rrg_lv001_cls_lv003"
    display_name = "F1HopprCthRrgLv001ClsLv003"

    def __init__(self, **kwargs):
        from .f1hoppr_cth_rrg_lv001_cls_lv003 import HopprF1CthRrgLv001ClsLv003
        if HopprF1CthRrgLv001ClsLv003 is None:
            raise ImportError(
                "HopprF1CthRrgLv001ClsLv003 failed to import — missing dependency "
                "or checkpoint. See "
                "radeval/metrics/f1hoppr_cth_rrg_lv001_cls_lv003/__init__.py.")
        self._scorer = HopprF1CthRrgLv001ClsLv003()

    def metric_keys(self, detailed=False):
        return [
            "f1hoppr_cth_rrg_lv001_cls_lv003_accuracy",
            "f1hoppr_cth_rrg_lv001_cls_lv003_micro_f1",
            "f1hoppr_cth_rrg_lv001_cls_lv003_macro_f1",
            "f1hoppr_cth_rrg_lv001_cls_lv003_weighted_f1",
        ]

    def progress_total(self, n):
        import math
        bs = self._scorer.batch_size
        return 2 * math.ceil(n / bs)  # one pass over hyps + one over refs

    def compute(self, refs, hyps, per_sample=False, detailed=False,
                on_progress=None):
        """Override: per_sample mode returns different keys than default."""
        accuracy, sample_acc, report = self._scorer(
            hyps, refs, on_batch_done=on_progress)
        k = "f1hoppr_cth_rrg_lv001_cls_lv003"

        if per_sample:
            return {
                f"{k}_sample_acc": (
                    sample_acc.tolist() if hasattr(sample_acc, 'tolist')
                    else list(sample_acc)),
            }
        elif detailed:
            labels = {n: v["f1-score"] for n, v in list(report.items())[:-4]}
            return {
                f"{k}_accuracy": round(accuracy, 4),
                f"{k}_micro_f1": round(report["micro avg"]["f1-score"], 4),
                f"{k}_macro_f1": round(report["macro avg"]["f1-score"], 4),
                f"{k}_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
                f"{k}_label_scores_f1": labels,
            }
        else:
            return {
                f"{k}_accuracy": round(accuracy, 4),
                f"{k}_micro_f1": round(report["micro avg"]["f1-score"], 4),
                f"{k}_macro_f1": round(report["macro avg"]["f1-score"], 4),
                f"{k}_weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            }
