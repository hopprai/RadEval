"""Tests for the CTB P0/core10k/all f1hoppr metric family."""
import os

import pytest

from radeval.metrics.f1hoppr_ctb_rrg_lv001_cls_lv001_all import (
    HopprF1CtbRrgLv001ClsLv001All,
)
from radeval.metrics.f1hoppr_ctb_rrg_lv001_cls_lv001_core10k import (
    HopprF1CtbRrgLv001ClsLv001Core10k,
)
from radeval.metrics.f1hoppr_ctb_rrg_lv001_cls_lv001_p0 import (
    HopprF1CtbRrgLv001ClsLv001P0,
)

PREFIX = "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints"
SCOPES = {
    "p0": (
        HopprF1CtbRrgLv001ClsLv001P0,
        "f1hoppr_ctb_rrg_lv001_cls_lv001_p0",
        9,
    ),
    "core10k": (
        HopprF1CtbRrgLv001ClsLv001Core10k,
        "f1hoppr_ctb_rrg_lv001_cls_lv001_core10k",
        47,
    ),
    "all": (
        HopprF1CtbRrgLv001ClsLv001All,
        "f1hoppr_ctb_rrg_lv001_cls_lv001_all",
        87,
    ),
}
if any(cls is None or not os.path.isdir(f"{PREFIX}/{name}") for cls, name, _ in SCOPES.values()):
    pytest.skip("CTB f1hoppr metric/checkpoint unavailable", allow_module_level=True)

GALLSTONE = "Gallbladder / Biliary: There may be a tiny 2 mm gallstone noted."
BOWEL_OBSTRUCTION = (
    "Lower Chest: Small hiatal hernia. Gastrointestinal: Dilated small bowel "
    "measuring up to 3.6 cm with fecal distention, consistent with bowel "
    "obstruction. The area of transition is in the right lower quadrant."
)
NORMAL = "No acute abnormality in the abdomen or pelvis."
REPORTS = [GALLSTONE, BOWEL_OBSTRUCTION, NORMAL]


@pytest.fixture(scope="module", params=list(SCOPES))
def scorer_info(request):
    cls, name, n_conditions = SCOPES[request.param]
    return request.param, name, n_conditions, cls()


def test_output_shape_and_condition_order(scorer_info):
    _, _, n_conditions, scorer = scorer_info
    matrix = scorer._predict_label_matrix(REPORTS)
    assert matrix.shape == (3, n_conditions + 1)
    assert scorer.LABELS == scorer.model.condition_names


def test_real_reports(scorer_info):
    scope, _, _, scorer = scorer_info
    matrix = scorer._predict_label_matrix(REPORTS)
    predicted = [
        {scorer.LABELS[idx] for idx, value in enumerate(row[:-1]) if value}
        for row in matrix
    ]
    if scope == "p0":
        assert predicted[0] == set()  # gallstones are outside the 9-head scope
    else:
        assert predicted[0] == {"cholelithiasis_or_gallstones"}
    assert "bowel_obstruction" in predicted[1]
    assert predicted[2] == set()
    assert matrix[2, -1] == 1


def test_identical_reports_score_perfectly(scorer_info):
    _, _, _, scorer = scorer_info
    accuracy, sample_acc, report = scorer(REPORTS, REPORTS)
    assert accuracy == 1.0
    assert sample_acc == [1.0, 1.0, 1.0]
    assert "micro avg" in report


def test_radeval_default_and_detailed(scorer_info):
    from radeval import RadEval

    _, name, _, _ = scorer_info
    default = RadEval(metrics=[name], show_progress=False)
    assert default(refs=REPORTS, hyps=REPORTS)[f"{name}_accuracy"] == 1.0
    detailed = RadEval(metrics=[name], detailed=True, show_progress=False)
    result = detailed(refs=REPORTS, hyps=REPORTS)
    assert f"{name}_label_scores_f1" in result
