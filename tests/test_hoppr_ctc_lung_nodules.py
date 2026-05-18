"""Integration tests for the hoppr_ctc_lung_nodules metric.

Uses mocked LLM responses so tests are deterministic and don't hit the API.
Four scenarios:
    1. Perfect match (ref == hyp, one nodule)
    2. Size tolerance error (8 mm vs 20 mm — 12 mm delta exceeds 15% of 8 mm)
    3. Size exact-match-only difference (8 mm vs 9 mm — within 1.25 mm
       tolerance for the 6-8 mm bucket but not exactly equal)
    4. Complete miss (ref has a nodule, hyp has none — different section
       entirely)

Run with:
    pytest tests/test_hoppr_ctc_lung_nodules.py -s
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore

if HopprCTCLungNodulesScore is None:
    pytest.skip("HopprCTCLungNodulesScore not available", allow_module_level=True)


# Clean-findings fragments used in the tests. Must start with PULMONARY NODULES:
# per our schema (lv004/lv005).
_CF_WITH_8MM = (
    "PULMONARY NODULES: There is an 8 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_WITH_9MM = (
    "PULMONARY NODULES: There is a 9 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_WITH_20MM = (
    "PULMONARY NODULES: There is a 20 mm solid nodule in the right upper lobe. "
    "LUNGS AND AIRWAYS: No consolidation."
)
_CF_NO_NODULE = "LUNGS AND AIRWAYS: No consolidation. No pleural effusion."


def _make_nodule(prefix, idx, size, bucket, location="right upper lobe",
                 laterality="right", density="solid", calcified=False,
                 noun="nodule"):
    return {
        "id": f"{prefix}{idx}",
        "size_mm": size,
        "size_bucket": bucket,
        "density": density,
        "calcified": calcified,
        "location": location,
        "laterality": laterality,
        "noun": noun,
        "uncertain": False,
        "text": f"There is a {size} mm {density} {noun} in the {location}.",
    }


def _make_pair(rid, pid, ref_size, pred_size, ref_bucket, pred_bucket,
               size_error=False, size_exact_match=False,
               density_error=False, calcified_error=False,
               location_error=False, noun_error=False,
               uncertainty_error=False):
    return {
        "ref_id": rid, "pred_id": pid,
        "ref_size_mm": ref_size, "pred_size_mm": pred_size,
        "ref_size_bucket": ref_bucket, "pred_size_bucket": pred_bucket,
        "size_error": size_error, "size_exact_match": size_exact_match,
        "density_error": density_error, "calcified_error": calcified_error,
        "location_error": location_error, "noun_error": noun_error,
        "uncertainty_error": uncertainty_error,
        "notes": "",
    }


_RESP_PERFECT = {
    "reference_nodules":  [_make_nodule("R", 1, 8, "6to8")],
    "predicted_nodules":  [_make_nodule("P", 1, 8, "6to8")],
    "matched_pairs": [_make_pair("R1", "P1", 8, 8, "6to8", "6to8",
                                 size_exact_match=True)],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_SIZE_TOLERANCE_ERR = {
    "reference_nodules":  [_make_nodule("R", 1, 8, "6to8")],
    "predicted_nodules":  [_make_nodule("P", 1, 20, "gt15")],
    "matched_pairs": [_make_pair("R1", "P1", 8, 20, "6to8", "gt15",
                                 size_error=True)],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_SIZE_INEXACT_ONLY = {
    "reference_nodules":  [_make_nodule("R", 1, 8, "6to8")],
    "predicted_nodules":  [_make_nodule("P", 1, 9, "8to15")],
    "matched_pairs": [_make_pair("R1", "P1", 8, 9, "6to8", "8to15")],
    "false_findings": [],
    "missing_findings": [],
}

_RESP_COMPLETE_MISS = {
    "reference_nodules":  [_make_nodule("R", 1, 8, "6to8")],
    "predicted_nodules":  [],
    "matched_pairs": [],
    "false_findings": [],
    "missing_findings": ["R1"],
}


def _mock_openai_response(content):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 100
    response.usage.total_tokens = 200
    return response


class TestUnit:
    """Pure-Python tests on the utils (no LLM / no imports of the scorer)."""

    def test_extract_pn_segment_present(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import extract_pn_segment
        cf = (
            "LUNGS AND AIRWAYS: Clear. "
            "PULMONARY NODULES: There is an 8 mm solid nodule in the right upper lobe. "
            "MEDIASTINUM: No adenopathy."
        )
        assert (extract_pn_segment(cf)
                == "There is an 8 mm solid nodule in the right upper lobe.")

    def test_extract_pn_segment_at_start(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import extract_pn_segment
        cf = (
            "PULMONARY NODULES: There is a 5 mm nodule in the lingula. "
            "LUNGS AND AIRWAYS: Emphysema."
        )
        assert extract_pn_segment(cf) == "There is a 5 mm nodule in the lingula."

    def test_extract_pn_segment_absent(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import extract_pn_segment
        cf = "LUNGS AND AIRWAYS: Clear. MEDIASTINUM: No adenopathy."
        assert extract_pn_segment(cf) == ""

    def test_scoring_perfect_match(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_PERFECT)
        assert m["detection_f1"] == 1.0
        assert m["size_accuracy"] == 1.0
        assert m["size_exact_match"] == 1.0
        assert m["size_mae_mm"] == 0.0
        assert m["size_mape"] == 0.0
        assert m["composite"] == 1.0
        # New per-bucket counters: 1 TP in 6to8, 0 FN, 0 FP everywhere.
        assert m["tp_6to8"] == 1
        assert m["fn_6to8"] == 0
        assert m["fp_6to8"] == 0
        # Density two-class: ref/pred both solid -> 1/1 solid.
        assert m["density_correct_solid"] == 1
        assert m["density_total_solid"] == 1
        assert m["density_accuracy_solid"] == 1.0

    def test_scoring_size_tolerance_error(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_SIZE_TOLERANCE_ERR)
        assert m["detection_f1"] == 1.0       # matched 1/1, no false/miss
        assert m["size_accuracy"] == 0.0      # outside tolerance
        assert m["size_exact_match"] == 0.0
        assert m["size_mae_mm"] == 12.0
        assert abs(m["size_mape"] - 1.5) < 1e-9
        # 1 matched with 1 attr error (size): credit = 1 / (1 + 0.5) = 0.6667
        assert abs(m["composite"] - 0.6667) < 0.01
        # Per-bucket: TP recorded in ref bucket 6to8.
        assert m["tp_6to8"] == 1
        # abs error 12 mm in the 6to8 bucket.
        assert m["abs_err_n_6to8"] == 1
        assert m["abs_err_sum_6to8"] == 12.0

    def test_scoring_size_exact_match_only(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_SIZE_INEXACT_ONLY)
        assert m["detection_f1"] == 1.0
        assert m["size_accuracy"] == 1.0   # within tolerance (1 mm <= 1.25 mm)
        assert m["size_exact_match"] == 0.0
        assert m["size_mae_mm"] == 1.0
        assert abs(m["size_mape"] - 0.125) < 1e-9
        assert m["composite"] == 1.0
        # Diameter MAE accumulator for the ref bucket 6to8.
        assert m["abs_err_n_6to8"] == 1
        assert m["abs_err_sum_6to8"] == 1.0

    def test_scoring_complete_miss(self):
        from radeval.metrics.hoppr_ctc_lung_nodules.utils import compute_per_row_metrics
        m = compute_per_row_metrics(_RESP_COMPLETE_MISS)
        assert m["detection_recall"] == 0.0
        assert m["detection_precision"] is None
        assert m["detection_f1"] is None
        assert m["size_accuracy"] is None   # no matched pairs
        assert m["composite"] == 0.0
        # Missed ref nodule should land in fn_6to8.
        assert m["fn_6to8"] == 1
        assert m["tp_6to8"] == 0


class TestIntegration:
    """Integration tests with a mocked OpenAI client."""

    @pytest.fixture
    def mock_openai_client(self):
        with patch("openai.OpenAI") as mock_class, \
             patch("openai.AsyncOpenAI") as mock_async_class:
            mock_sync = MagicMock()
            mock_class.return_value = mock_sync
            mock_async = MagicMock()
            mock_async_class.return_value = mock_async
            yield mock_sync, mock_async

    def test_import(self):
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        from radeval.metrics.hoppr_ctc_lung_nodules.adapter import HopprCTCLungNodulesMetric
        assert HopprCTCLungNodulesScore is not None
        assert HopprCTCLungNodulesMetric is not None

    def test_invalid_provider(self):
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        with pytest.raises(NotImplementedError, match="does not support"):
            HopprCTCLungNodulesScore(provider="invalid", openai_api_key="x")

    def test_initialization_with_api_key(self, mock_openai_client):
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        scorer = HopprCTCLungNodulesScore(
            provider="openai", openai_api_key="test-key")
        assert scorer.provider == "openai"
        assert scorer.model_name == HopprCTCLungNodulesScore.DEFAULT_OPENAI_MODEL

    def test_both_empty_short_circuit(self, mock_openai_client):
        """Rows with no PN section on either side should skip the LLM call."""
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        scorer = HopprCTCLungNodulesScore(provider="openai", openai_api_key="x")

        refs = [_CF_NO_NODULE]
        hyps = [_CF_NO_NODULE]

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("{}")
        )

        mean, std, per_sample, df = scorer(refs, hyps)
        assert per_sample[0] == 1.0   # both-empty -> composite=1.0
        mock_async.chat.completions.create.assert_not_called()

    def test_full_pipeline_perfect_match(self, mock_openai_client):
        """End-to-end with mocked LLM returning the perfect-match JSON."""
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        scorer = HopprCTCLungNodulesScore(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        mean, std, per_sample, df = scorer([_CF_WITH_8MM], [_CF_WITH_8MM])
        assert per_sample[0] == 1.0
        assert mean == 1.0
        assert df.iloc[0]["detection_f1"] == 1.0
        assert df.iloc[0]["size_exact_match"] == 1.0
        assert df.iloc[0]["tp_6to8"] == 1

    def test_full_pipeline_size_tolerance_err(self, mock_openai_client):
        from radeval.metrics.hoppr_ctc_lung_nodules import HopprCTCLungNodulesScore
        scorer = HopprCTCLungNodulesScore(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_SIZE_TOLERANCE_ERR))
        )

        mean, std, per_sample, df = scorer([_CF_WITH_8MM], [_CF_WITH_20MM])
        assert abs(per_sample[0] - 0.6667) < 0.01
        assert df.iloc[0]["size_mae_mm"] == 12.0
        assert df.iloc[0]["size_accuracy"] == 0.0

    def test_adapter_default_mode(self, mock_openai_client):
        """Verify HopprCTCLungNodulesMetric.compute() returns aggregate keys."""
        from radeval.metrics.hoppr_ctc_lung_nodules.adapter import HopprCTCLungNodulesMetric
        metric = HopprCTCLungNodulesMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=False, detailed=False)
        assert "hoppr_ctc_lung_nodules_detection_f1" in out
        assert "hoppr_ctc_lung_nodules_size_mae_mm" in out
        assert out["hoppr_ctc_lung_nodules_composite"] == 1.0
        # Per-bucket KPIs are surfaced by the adapter.
        assert "hoppr_ctc_lung_nodules_sensitivity_6to8" in out
        assert out["hoppr_ctc_lung_nodules_sensitivity_6to8"] == 1.0
        assert out["hoppr_ctc_lung_nodules_fp_per_study_6to8"] == 0.0

    def test_adapter_per_sample_mode(self, mock_openai_client):
        from radeval.metrics.hoppr_ctc_lung_nodules.adapter import HopprCTCLungNodulesMetric
        metric = HopprCTCLungNodulesMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=True, detailed=False)
        assert isinstance(out["hoppr_ctc_lung_nodules_composite"], list)
        assert out["hoppr_ctc_lung_nodules_composite"] == [1.0]

    def test_adapter_detailed_mode(self, mock_openai_client):
        from radeval.metrics.hoppr_ctc_lung_nodules.adapter import HopprCTCLungNodulesMetric
        metric = HopprCTCLungNodulesMetric(provider="openai", openai_api_key="x")

        mock_sync, mock_async = mock_openai_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(json.dumps(_RESP_PERFECT))
        )

        out = metric.compute([_CF_WITH_8MM], [_CF_WITH_8MM],
                             per_sample=False, detailed=True)
        assert "hoppr_ctc_lung_nodules_composite" in out
        assert "hoppr_ctc_lung_nodules_composite_std" in out
        # Desiderata flags appear in detailed mode.
        assert "hoppr_ctc_lung_nodules_sensitivity_6to8_pass" in out
        assert "hoppr_ctc_lung_nodules_desiderata_pass" in out
