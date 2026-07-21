import inspect

import pytest

from radeval.metrics._registry import get_metric_class


# GPU metrics whose adapters now expose a tunable batch_size, with the default
# each preserves (== the scorer's prior hardcoded value).
BATCH_SIZE_METRICS = {
    "bertscore": 64,
    "f1chexbert": 64,
    "f1radbert_ct": 16,
    "srrbert": 4,
    "ratescore": 1,
    "green": 8,
}


@pytest.mark.parametrize("name,default", sorted(BATCH_SIZE_METRICS.items()))
def test_adapter_exposes_batch_size(name, default):
    sig = inspect.signature(get_metric_class(name).__init__)
    assert "batch_size" in sig.parameters, f"{name} adapter missing batch_size param"
    assert sig.parameters["batch_size"].default == default


def test_green_scorer_ctor_accepts_batch_size():
    from radeval.metrics.green_score.green import GREEN
    sig = inspect.signature(GREEN.__init__)
    assert "batch_size" in sig.parameters
    assert sig.parameters["batch_size"].default == 8
