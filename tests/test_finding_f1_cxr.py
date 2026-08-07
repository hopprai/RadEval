"""Test finding_f1_cxr metric."""
import os
import pytest


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_finding_f1_cxr_basic():
    """Test basic finding F1 extraction on small examples."""
    from radeval import RadEval

    refs = [
        "The lungs are clear. No pleural effusion or pneumothorax.",
        "There is a moderate left-sided pleural effusion. Endotracheal tube is in place.",
    ]
    hyps = [
        "Clear lungs bilaterally. No effusion or pneumothorax identified.",
        "Left pleural effusion is noted. ETT is present.",
    ]

    evaluator = RadEval(metrics=["finding_f1_cxr"])
    results = evaluator(refs=refs, hyps=hyps)

    # Check that we get the expected output keys
    assert "finding_f1" in results
    assert "finding_precision" in results
    assert "finding_recall" in results
    assert "macro_f1" in results
    assert "micro_f1" in results
    assert "micro_sensitivity" in results
    assert "micro_specificity" in results
    assert "micro_ppv" in results
    assert "micro_npv" in results

    # Basic sanity checks
    assert 0.0 <= results["finding_f1"] <= 1.0
    assert 0.0 <= results["micro_f1"] <= 1.0
    assert 0.0 <= results["micro_sensitivity"] <= 1.0
    assert 0.0 <= results["micro_specificity"] <= 1.0

    print(f"✓ finding_f1: {results['finding_f1']:.3f}")
    print(f"✓ micro_f1: {results['micro_f1']:.3f}")
    print(f"✓ micro_sensitivity: {results['micro_sensitivity']:.3f}")
    print(f"✓ micro_ppv: {results['micro_ppv']:.3f}")


if __name__ == "__main__":
    test_finding_f1_cxr_basic()
