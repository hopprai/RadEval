"""Tests for HopprF1MskRrgLv005ClsLv009 using real full-body MSK report examples.

The MSK classifier uses 3-way coding per finding:
  0 = absent or not reported, 1 = present, 2 = uncertain
Binary mapping: {0} -> negative, {1, 2} -> positive (present or uncertain).
"""
import os
import pytest

from radeval.metrics.f1hoppr_msk_rrg_lv005_cls_lv009 import HopprF1MskRrgLv005ClsLv009

_CKPT_DIR = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/"
    "f1hoppr_msk_rrg_lv005_cls_lv009"
)

if HopprF1MskRrgLv005ClsLv009 is None or not os.path.isdir(_CKPT_DIR):
    pytest.skip(
        "HopprF1MskRrgLv005ClsLv009 not available (missing module or checkpoint)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def scorer():
    return HopprF1MskRrgLv005ClsLv009(checkpoint_dir=_CKPT_DIR)


# ── Real examples from the dataset ───────────────────────────────────────
# Finding order (26 heads): acute_fracture, healed_fracture, osteoarthritis,
# osteophyte_formation, joint_space_narrowing, subchondral_change,
# degenerative_spine_disease, disc_space_narrowing, foraminal_narrowing,
# spondylolisthesis, scoliosis, abnormal_spinal_curvature,
# joint_subluxation_dislocation, joint_effusion, soft_tissue_swelling,
# soft_tissue_calcification, osteopenia, bone_lesion, erosion, hardware_present,
# arthroplasty, spinal_fusion, avascular_necrosis, chondral_lesion,
# accessory_ossicle, spine_developmental_variant

SAMPLE_OA_OSTEOPHYTE = (
    "Joints: Mild tricompartmental productive changes with spiking of the tibial "
    "spines are seen consistent with mild osteoarthritis."
)
GT_OA_OSTEOPHYTE = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0]

SAMPLE_ACUTE_FX = "Bones: Acute nondisplaced fracture of base of fifth metatarsal."
GT_ACUTE_FX = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0]

SAMPLE_NORMAL = "No acute osseous abnormality."
GT_NORMAL = [0] * 26

REAL_REFS = [SAMPLE_OA_OSTEOPHYTE, SAMPLE_ACUTE_FX, SAMPLE_NORMAL]
REAL_GT_BINARY = [GT_OA_OSTEOPHYTE, GT_ACUTE_FX, GT_NORMAL]


# ── Tests: model output structure ────────────────────────────────────────

class TestHopprF1MskDirect:

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

class TestHopprF1MskPredictions:
    """Verify the model correctly identifies findings from real MSK reports."""

    def test_osteoarthritis_and_osteophyte(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_OA_OSTEOPHYTE])
        pred_binary = y_pred[0, :26].tolist()
        assert pred_binary == GT_OA_OSTEOPHYTE, (
            f"Expected {GT_OA_OSTEOPHYTE}, got {pred_binary}")

    def test_acute_fracture(self, scorer):
        y_pred = scorer._predict_label_matrix([SAMPLE_ACUTE_FX])
        pred_binary = y_pred[0, :26].tolist()
        assert pred_binary == GT_ACUTE_FX, (
            f"Expected {GT_ACUTE_FX}, got {pred_binary}")

    def test_no_finding_column(self, scorer):
        """When all 26 findings are negative, no_finding should be 1."""
        y_pred = scorer._predict_label_matrix([SAMPLE_NORMAL])
        all_negative = y_pred[0, :26].sum() == 0
        no_finding = y_pred[0, 26]
        if all_negative:
            assert no_finding == 1


# ── Tests: RadEval integration ───────────────────────────────────────────

class TestHopprF1MskViaRadEval:

    def test_basic_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_msk_rrg_lv005_cls_lv009"], show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_msk_rrg_lv005_cls_lv009_accuracy" in results
        assert results["f1hoppr_msk_rrg_lv005_cls_lv009_accuracy"] == 1.0

    def test_details_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_msk_rrg_lv005_cls_lv009"], detailed=True,
            show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_msk_rrg_lv005_cls_lv009_accuracy" in results
        assert "f1hoppr_msk_rrg_lv005_cls_lv009_label_scores_f1" in results
        assert isinstance(
            results["f1hoppr_msk_rrg_lv005_cls_lv009_label_scores_f1"], dict)

    def test_per_sample_output(self):
        from radeval import RadEval
        evaluator = RadEval(
            metrics=["f1hoppr_msk_rrg_lv005_cls_lv009"], per_sample=True,
            show_progress=False)
        results = evaluator(refs=REAL_REFS, hyps=REAL_REFS)
        assert "f1hoppr_msk_rrg_lv005_cls_lv009_sample_acc" in results
        assert isinstance(
            results["f1hoppr_msk_rrg_lv005_cls_lv009_sample_acc"], list)
        assert len(results["f1hoppr_msk_rrg_lv005_cls_lv009_sample_acc"]) == len(REAL_REFS)
        assert all(s == 1.0 for s in
                   results["f1hoppr_msk_rrg_lv005_cls_lv009_sample_acc"])
