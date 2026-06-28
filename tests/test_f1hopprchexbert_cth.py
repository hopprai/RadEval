"""Tests for HopprF1CheXbertCTH using real CT-head report examples.

The CTH classifier uses 3-way coding per finding:
  0 = absent or not reported, 1 = present, 2 = uncertain
Binary mapping: {0} -> negative, {1, 2} -> positive (present or uncertain).
"""
import os
import pytest

from radeval.metrics.f1hopprchexbert_cth import HopprF1CheXbertCTH

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hopprchexbert_cth"
)

if HopprF1CheXbertCTH is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1CheXbertCTH not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1CheXbertCTH(checkpoint_dir=_CKPT_DIR)


# ── Real examples from the dataset ───────────────────────────────────────
# Finding order (10 heads):
#   0  atrophy
#   1  white_matter_disease
#   2  sinus_disease
#   3  post_surgical_calvarium
#   4  intracranial_atherosclerosis
#   5  lens_replacement
#   6  scleral_buckle
#   7  technical_limitation
#   8  encephalomalacia
#   9  mastoid_effusion

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
REAL_GT_BINARY = [GT_BINARY_ATROPHY_WMD, GT_BINARY_SINUS, GT_BINARY_NORMAL]


# ── Tests: model output structure ────────────────────────────────────────

class TestHopprF1CheXbertCTHDirect:

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


# ── Tests: model predictions match ground truth ──────────────────────────

class TestHopprF1CheXbertCTHPredictions:
    """Verify the model correctly identifies findings from real CT-head reports."""

    def test_atrophy_and_white_matter_disease(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_ATROPHY_WMD])
        pred_binary = y_pred[0, :10].tolist()
        assert pred_binary == GT_BINARY_ATROPHY_WMD, (
            f"Expected {GT_BINARY_ATROPHY_WMD}, got {pred_binary}")

    def test_sinus_disease(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_SINUS])
        pred_binary = y_pred[0, :10].tolist()
        assert pred_binary == GT_BINARY_SINUS, (
            f"Expected {GT_BINARY_SINUS}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 10 findings are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix([SAMPLE_NORMAL])
        all_negative = y_pred[0, :10].sum() == 0
        no_finding = y_pred[0, 10]
        if all_negative:
            assert no_finding == 1


# ── Tests: RadEval integration ───────────────────────────────────────────

class TestHopprF1CheXbertCTHViaRadEval:

    def test_basic_output(self):
        from radeval import RadEval
        evaluator = RadEval(metrics=["f1hopprchexbert_cth"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_cth_accuracy" in results
        assert results["f1hopprchexbert_cth_accuracy"] == 1.0

    def test_details_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_cth"], detailed=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hopprchexbert_cth_accuracy" in results
        assert "f1hopprchexbert_cth_label_scores_f1" in results
        assert isinstance(results["f1hopprchexbert_cth_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hopprchexbert_cth"], per_sample=True, show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)

        assert "f1hopprchexbert_cth_sample_acc" in results
        assert isinstance(results["f1hopprchexbert_cth_sample_acc"], list)
        assert len(results["f1hopprchexbert_cth_sample_acc"]) == len(REAL_REFS)

        assert all(s == 1.0 for s in results["f1hopprchexbert_cth_sample_acc"])

        assert "f1hopprchexbert_cth" not in results
        assert "f1hopprchexbert_cth_accuracy" not in results
