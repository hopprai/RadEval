from .._base import MetricBase


class F1HopprCheXbertV2Metric(MetricBase):
    name = "f1hopprchexbert_v2"
    display_name = "F1HopprCheXbertV2"

    def __init__(self, **kwargs):
        from .f1hopprchexbert_v2 import HopprF1CheXbertV2
        self._scorer = HopprF1CheXbertV2()

    def metric_keys(self, detailed=False):
        return [
            "f1hopprchexbert_v2_5_micro_f1", "f1hopprchexbert_v2_all_micro_f1",
            "f1hopprchexbert_v2_5_macro_f1", "f1hopprchexbert_v2_all_macro_f1",
            "f1hopprchexbert_v2_5_weighted_f1", "f1hopprchexbert_v2_all_weighted_f1",
        ]

    def progress_total(self, n):
        import math
        return 2 * math.ceil(n / self._scorer.batch_size)

    def compute(self, refs, hyps, per_sample=False, detailed=False, on_progress=None):
        _, _, cr_all, cr_5, sample_acc_full, sample_acc_5 = \
            self._scorer.forward(hyps, refs, on_batch_done=on_progress)

        if per_sample:
            return {
                "f1hopprchexbert_v2_sample_acc_5": list(sample_acc_5),
                "f1hopprchexbert_v2_sample_acc_all": list(sample_acc_full),
            }

        return {
            "f1hopprchexbert_v2_5_micro_f1": round(cr_5["micro avg"]["f1-score"], 4),
            "f1hopprchexbert_v2_all_micro_f1": round(cr_all["micro avg"]["f1-score"], 4),
            "f1hopprchexbert_v2_5_macro_f1": round(cr_5["macro avg"]["f1-score"], 4),
            "f1hopprchexbert_v2_all_macro_f1": round(cr_all["macro avg"]["f1-score"], 4),
            "f1hopprchexbert_v2_5_weighted_f1": round(cr_5["weighted avg"]["f1-score"], 4),
            "f1hopprchexbert_v2_all_weighted_f1": round(cr_all["weighted avg"]["f1-score"], 4),
        }
