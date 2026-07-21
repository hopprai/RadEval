"""Shared MetricBase adapter for private f1hoppr_* multi-output scorers."""
from .._base import MetricBase


class F1HopprMetricAdapter(MetricBase):
    scorer_class = None

    def __init__(self, **kwargs):
        if self.scorer_class is None:
            raise TypeError("Subclasses must define scorer_class")
        self._scorer = self.scorer_class(**kwargs)

    def metric_keys(self, detailed=False):
        return [
            f"{self.name}_accuracy",
            f"{self.name}_micro_f1",
            f"{self.name}_macro_f1",
            f"{self.name}_weighted_f1",
        ]

    def progress_total(self, n):
        import math

        return 2 * math.ceil(n / self._scorer.batch_size)

    def compute(
        self,
        refs,
        hyps,
        per_sample=False,
        detailed=False,
        on_progress=None,
    ):
        accuracy, sample_acc, report = self._scorer(
            hyps, refs, on_batch_done=on_progress
        )
        if per_sample:
            return {f"{self.name}_sample_acc": list(sample_acc)}
        result = {
            f"{self.name}_accuracy": round(accuracy, 4),
            f"{self.name}_micro_f1": round(report["micro avg"]["f1-score"], 4),
            f"{self.name}_macro_f1": round(report["macro avg"]["f1-score"], 4),
            f"{self.name}_weighted_f1": round(
                report["weighted avg"]["f1-score"], 4
            ),
        }
        if detailed:
            result[f"{self.name}_label_scores_f1"] = {
                key: value["f1-score"]
                for key, value in list(report.items())[:-4]
            }
        return result
