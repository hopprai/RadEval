"""Tests for HopprF1CheXbertMammo using real mammography report examples.

The model classifies 5 mammography conditions, each with 4-way coding:
  0 = definitely absent, 1 = not reported, 2 = uncertain, 3 = definitely present
Binary mapping: {0,1} -> negative, {2,3} -> positive.
"""
import os
import pytest

from radeval.metrics.f1hopprchexbert_mammo import HopprF1CheXbertMammo

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hopprchexbert_mammo"
)

if HopprF1CheXbertMammo is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertMammo not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertMammo(checkpoint_dir=_CKPT_DIR)


# -- Real examples ----------------------------------------------------------
# Condition order (5 heads, matches training __init__.py):
#   0  mass
#   1  calcifications
#   2  biopsy_marker
#   3  breast_implant
#   4  pacemaker

SAMPLE_MASS = (
    "There is an irregular spiculated mass in the upper outer quadrant of the "
    "left breast measuring 1.5 cm, suspicious for malignancy."
)
GT_BINARY_MASS = [1, 0, 0, 0, 0]

SAMPLE_CALCIFICATIONS = (
    "Scattered pleomorphic microcalcifications are noted in the right breast "
    "in a segmental distribution."
)
GT_BINARY_CALCIFICATIONS = [0, 1, 0, 0, 0]

SAMPLE_BIOPSY_MARKER = (
    "A biopsy clip marker is present in the left breast at the site of prior "
    "stereotactic biopsy."
)
GT_BINARY_BIOPSY_MARKER = [0, 0, 1, 0, 0]

SAMPLE_IMPLANT = (
    "Bilateral subpectoral breast implants are intact without evidence of "
    "rupture."
)
GT_BINARY_IMPLANT = [0, 0, 0, 1, 0]

SAMPLE_PACEMAKER = (
    "A cardiac pacemaker device is noted overlying the left chest wall."
)
GT_BINARY_PACEMAKER = [0, 0, 0, 0, 1]

SAMPLE_NORMAL = (
    "The breast tissue is heterogeneously dense. No suspicious masses, "
    "calcifications, or architectural distortion."
)
GT_BINARY_NORMAL = [0, 0, 0, 0, 0]

REAL_REFS = [
    SAMPLE_MASS,
    SAMPLE_CALCIFICATIONS,
    SAMPLE_BIOPSY_MARKER,
    SAMPLE_IMPLANT,
    SAMPLE_PACEMAKER,
]


# -- Tests: model output structure ------------------------------------------

class TestHopprF1CheXbertMammoDirect:

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

    def test_report_has_aggregate_keys(self, scorer):
        _, _, report = scorer(REAL_REFS, REAL_REFS)
        for key in ("micro avg", "macro avg", "weighted avg"):
            assert key in report
            for field in ("precision", "recall", "f1-score", "support"):
                assert field in report[key]

    def test_identical_reports_perfect_accuracy(self, scorer):
        accuracy, per_sample, _ = scorer(REAL_REFS, REAL_REFS)
        assert accuracy == 1.0
        assert all(s == 1.0 for s in per_sample)

    def test_validation_errors(self, scorer):
        with pytest.raises(TypeError):
            scorer("not a list", REAL_REFS)
        with pytest.raises(ValueError):
            scorer(REAL_REFS[:1], REAL_REFS)


# -- Tests: model predictions match ground truth ----------------------------

class TestHopprF1CheXbertMammoPredictions:
    """Verify the model correctly identifies conditions from real reports."""

    def test_mass(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_MASS])
        pred_binary = y_pred[0, :5].tolist()
        assert pred_binary == GT_BINARY_MASS, (
            f"Expected {GT_BINARY_MASS}, got {pred_binary}")

    def test_calcifications(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_CALCIFICATIONS])
        pred_binary = y_pred[0, :5].tolist()
        assert pred_binary == GT_BINARY_CALCIFICATIONS, (
            f"Expected {GT_BINARY_CALCIFICATIONS}, got {pred_binary}")

    def test_biopsy_marker(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_BIOPSY_MARKER])
        pred_binary = y_pred[0, :5].tolist()
        assert pred_binary == GT_BINARY_BIOPSY_MARKER, (
            f"Expected {GT_BINARY_BIOPSY_MARKER}, got {pred_binary}")

    def test_implant(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_IMPLANT])
        pred_binary = y_pred[0, :5].tolist()
        assert pred_binary == GT_BINARY_IMPLANT, (
            f"Expected {GT_BINARY_IMPLANT}, got {pred_binary}")

    def test_pacemaker(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_PACEMAKER])
        pred_binary = y_pred[0, :5].tolist()
        assert pred_binary == GT_BINARY_PACEMAKER, (
            f"Expected {GT_BINARY_PACEMAKER}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 5 conditions are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix([SAMPLE_NORMAL])
        all_negative = y_pred[0, :5].sum() == 0
        no_finding = y_pred[0, 5]
        assert all_negative
        assert no_finding == 1


# -- Tests: RadEval integration ---------------------------------------------

class TestHopprF1CheXbertMammoViaRadEval:

    def test_basic_output(self):
        from radeval import RadEval
        evaluator = RadEval(metrics=["f1hopprchexbert_mammo"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_mammo_accuracy" in results
        assert results["f1hopprchexbert_mammo_accuracy"] == 1.0

    def test_details_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_mammo"], detailed=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_mammo_accuracy" in results
        assert "f1hopprchexbert_mammo_label_scores_f1" in results
        assert isinstance(results["f1hopprchexbert_mammo_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_mammo"], per_sample=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)

        assert "f1hopprchexbert_mammo_sample_acc" in results
        assert isinstance(results["f1hopprchexbert_mammo_sample_acc"], list)
        assert len(results["f1hopprchexbert_mammo_sample_acc"]) == len(REAL_REFS)

        assert all(s == 1.0 for s in results["f1hopprchexbert_mammo_sample_acc"])
