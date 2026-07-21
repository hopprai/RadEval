import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_f1Radbert_ct_exact_outputs():
    repo_id = "IAMJB/RadBERT-CT"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()

    texts = [
        "No acute cardiopulmonary abnormality.",
        "Right lower lobe opacity, suspicious for pneumonia. Pleural effusion present.",
    ]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        pred_mask = probs > 0.5

    expected_logits = torch.tensor(
        [
            [
                -7.0824,
                -8.3267,
                -8.4071,
                -8.7052,
                -7.7598,
                -7.3826,
                -8.6580,
                -7.1315,
                -8.0535,
                -5.8760,
                -6.7699,
                -6.4723,
                -8.1472,
                -7.7759,
                -8.0363,
                -6.9240,
                -8.0222,
                -7.3431,
            ],
            [
                -3.4248,
                -6.8445,
                -4.1796,
                -4.0292,
                -6.3926,
                -6.6850,
                -5.9953,
                -6.0609,
                -3.6582,
                -4.7553,
                0.2473,
                -6.0450,
                7.5655,
                -4.9244,
                -4.6517,
                -2.8377,
                -6.6161,
                -4.6961,
            ],
        ],
        dtype=logits.dtype,
    )

    assert logits.shape == torch.Size([2, 18])
    assert torch.allclose(logits, expected_logits, atol=1e-4)
    assert pred_mask.tolist() == [
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False],
    ]
    assert [[i for i, on in enumerate(row) if on] for row in pred_mask.tolist()] == [[], [10, 12]]


def test_f1Radbert_ct_per_sample_exposes_label_matrices():
    """per_sample mode returns per-study pred/true label matrices, and corpus
    micro/macro F1 recomputed from them matches RadEval's own detailed numbers
    (this is what lets a caller bootstrap a corpus-F1 CI)."""
    import numpy as np
    from sklearn.metrics import f1_score
    from radeval import RadEval

    refs = [
        "No acute cardiopulmonary abnormality.",
        "Right lower lobe opacity, suspicious for pneumonia. Pleural effusion present.",
        "Coronary artery calcification. Centrilobular emphysema. No pleural effusion.",
    ]
    hyps = [
        "No acute cardiopulmonary process.",
        "Pleural effusion. Right basilar opacity consistent with pneumonia.",
        "Aortic atherosclerosis with emphysema.",
    ]

    ps = RadEval(metrics=["f1radbert_ct"], per_sample=True, show_progress=False)(
        refs=refs, hyps=hyps)
    assert "f1radbert_ct_pred_labels" in ps
    assert "f1radbert_ct_true_labels" in ps
    y_pred = np.array(ps["f1radbert_ct_pred_labels"])
    y_true = np.array(ps["f1radbert_ct_true_labels"])
    assert y_pred.shape == y_true.shape == (len(refs), len(F1RadbertCT_LABELS) + 1)

    detailed = RadEval(metrics=["f1radbert_ct"], detailed=True, show_progress=False)(
        refs=refs, hyps=hyps)
    assert round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4) == \
        detailed["f1radbert_ct_micro_f1"]
    assert round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4) == \
        detailed["f1radbert_ct_macro_f1"]


F1RadbertCT_LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

