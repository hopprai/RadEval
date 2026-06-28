"""Tests for HopprF1CthRrgLv001ClsLv003 (convention-renamed copy of f1hopprchexbert_cth).

3-way coding per finding: 0 = absent or not reported, 1 = present, 2 = uncertain.
Binary mapping: {0} -> negative, {1, 2} -> positive (present or uncertain).
"""
import os
import pytest

from radeval.metrics.f1hoppr_cth_rrg_lv001_cls_lv003 import HopprF1CthRrgLv001ClsLv003

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "f1hoppr_cth_rrg_lv001_cls_lv003"
)

if HopprF1CthRrgLv001ClsLv003 is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CthRrgLv001ClsLv003 not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CthRrgLv001ClsLv003(checkpoint_dir=_CKPT_DIR)


# Finding order (10 heads): atrophy, white_matter_disease, sinus_disease,
# post_surgical_calvarium, intracranial_atherosclerosis, lens_replacement,
# scleral_buckle, technical_limitation, encephalomalacia, mastoid_effusion

SAMPLE_ATROPHY_WMD = (
    "ATROPHY: There is mild diffuse atrophy. "
    "WHITE MATTER DISEASE: Moderate periventricular ischemic white matter changes."
)
GT_BINARY_ATROPHY_WMD = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

SAMPLE_SINUS = (
    "SINUSES AND ORBITS: Large mucosal pseudocyst or polyp of the left maxillary sinus."
)
GT_BINARY_SINUS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

SAMPLE_NORMAL = "No significant intracranial findings."
GT_BINARY_NORMAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

REAL_REFS = [SAMPLE_ATROPHY_WMD, SAMPLE_SINUS, SAMPLE_NORMAL]


class TestHopprF1CthDirect:

    def test_returns_correct_tuple(self, scorer):
        accuracy, per_sample, report = scorer(REAL_REFS, REAL_REFS)
        assert isinstance(accuracy, float)
        assert isinstance(per_sample, list)
        assert len(per_sample) == len(REAL_REFS)
        assert isinstance(report, dict)

    def test_report_has_all_conditions(self, scorer):
        _, _, report = scorer(REAL_REFS, REAL_REFS)
        for label in scorer.LABELS:
            assert label in report, f"Missing condition: {label}"
        assert scorer.NO_FINDING in report

    def test_identical_reports_perfect_accuracy(self, scorer):
        accuracy, per_sample, _ = scorer(REAL_REFS, REAL_REFS)
        assert accuracy == 1.0
        assert all(s == 1.0 for s in per_sample)

    def test_validation_errors(self, scorer):
        with pytest.raises(TypeError):
            scorer("not a list", REAL_REFS)
        with pytest.raises(ValueError):
            scorer(REAL_REFS[:1], REAL_REFS)


class TestHopprF1CthPredictions:

    def test_atrophy_and_white_matter_disease(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_ATROPHY_WMD])
        assert y_pred[0, :10].tolist() == GT_BINARY_ATROPHY_WMD

    def test_sinus_disease(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_SINUS])
        assert y_pred[0, :10].tolist() == GT_BINARY_SINUS

    def test_no_finding_column(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_NORMAL])
        if y_pred[0, :10].sum() == 0:
            assert y_pred[0, 10] == 1


class TestHopprF1CthViaRadEval:

    def test_basic_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_cth_rrg_lv001_cls_lv003"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_cth_rrg_lv001_cls_lv003_accuracy" in results
        assert results["f1hoppr_cth_rrg_lv001_cls_lv003_accuracy"] == 1.0

    def test_details_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_cth_rrg_lv001_cls_lv003"], detailed=True,
            show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_cth_rrg_lv001_cls_lv003_label_scores_f1" in results
        assert isinstance(
            results["f1hoppr_cth_rrg_lv001_cls_lv003_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_cth_rrg_lv001_cls_lv003"], per_sample=True,
            show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_cth_rrg_lv001_cls_lv003_sample_acc" in results
        assert len(results["f1hoppr_cth_rrg_lv001_cls_lv003_sample_acc"]) == len(REAL_REFS)
        assert all(s == 1.0 for s in
                   results["f1hoppr_cth_rrg_lv001_cls_lv003_sample_acc"])
