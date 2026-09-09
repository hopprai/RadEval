"""HopprF1CheXbert v2: ModernBERT-base with 56 CXR finding heads."""
from __future__ import annotations

import os
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from transformers import ModernBertModel, AutoTokenizer

_DEFAULT_V2_CKPT = "/fss/amber_c/f1hopprchexbert_v2/checkpoints/best.pth"

FINDINGS_56 = [
    "acute_rib_fracture", "air_space_opacity", "aorta_calcifications",
    "aorta_dilated_dissection_rupture", "aorta_ectatic_tortuous_unfolded",
    "atelectasis", "breast_implant", "bronchiectasis", "bullous_disease",
    "calcified_pleural_plaques", "cardiomegaly", "central_venous_catheter",
    "clavicle_fracture", "diffuse_nodular_miliary_lesions",
    "elevated_hemidiaphragm", "endotracheal_tube", "enteric_tube",
    "ge_junction_hardware", "hiatus_hernia", "hilar_lymphadenopathy",
    "humerus_fracture", "hyperinflation", "implantable_electronic_device",
    "intercostal_drain", "interstitial_thickening", "kyphosis",
    "lung_nodule_or_mass", "mastectomy", "mediastinal_mass_widening",
    "non_acute_rib_fracture", "nonsurgical_internal_foreign_body",
    "pacemaker_electronic_cardiac_device_or_wires", "pectus_excavatum",
    "pleural_effusion", "pleural_masses", "pleural_thickening",
    "pneumomediastinum", "pneumothorax", "portal_venous_gas",
    "prosthetic_heart_valve", "pulmonary_arterial_catheter",
    "pulmonary_artery_enlargement",
    "pulmonary_congestion_pulmonary_venous_congestion",
    "reduced_lung_markings_hypoperfusion", "scapular_fracture", "scoliosis",
    "shoulder_dislocation", "shoulder_replacement", "spinal_fixation",
    "spinal_vertebral_fracture", "spondylopathy", "sternotomy_wires",
    "subcutaneous_emphysema", "subdiaphragmatic_gas_free_abdominal_gas",
    "tracheal_deviation", "underinflation",
]

TOP5 = ["cardiomegaly", "air_space_opacity", "atelectasis", "pleural_effusion", "pneumothorax"]

# Findings with test-set F1 < 0.3 — excluded from aggregate metrics as unreliable
EXCLUDED_FINDINGS = {
    "portal_venous_gas",
    "diffuse_nodular_miliary_lesions",
    "reduced_lung_markings_hypoperfusion",
    "subdiaphragmatic_gas_free_abdominal_gas",
    "calcified_pleural_plaques",
    "mediastinal_mass_widening",
    "pleural_masses",
}
RELIABLE_IDX = [i for i, f in enumerate(FINDINGS_56) if f not in EXCLUDED_FINDINGS]
RELIABLE_FINDINGS = [FINDINGS_56[i] for i in RELIABLE_IDX]


class _V2Labeler(nn.Module):
    def __init__(self, *, device, checkpoint_path: str = _DEFAULT_V2_CKPT):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.bert = ModernBertModel.from_pretrained("answerdotai/ModernBERT-base")
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(56)])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"v2 checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.load_state_dict(state, strict=True)
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        cls_hidden = self.dropout(out.last_hidden_state[:, 0])
        return [head(cls_hidden) for head in self.heads]


class HopprF1CheXbertV2:
    """56-finding CheXbert label extraction + F1 evaluation."""

    def __init__(
        self,
        *,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 64,
        checkpoint_path: str = _DEFAULT_V2_CKPT,
    ):
        device_obj = torch.device(device) if isinstance(device, str) else device
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA unavailable; falling back to CPU.")
            device_obj = torch.device("cpu")

        self.device = device_obj
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = _V2Labeler(device=device_obj, checkpoint_path=checkpoint_path).eval()
        self.top5_idx = [FINDINGS_56.index(n) for n in TOP5]

    @torch.no_grad()
    def get_labels(self, reports: Sequence[str], mode: str = "rrg") -> List[List[int]]:
        labels: List[List[int]] = []
        for i in range(0, len(reports), self.batch_size):
            batch = reports[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)
            logits = self.model(input_ids, attention_mask)
            preds = torch.stack([h.argmax(dim=1) for h in logits], dim=1)

            for row in preds.tolist():
                if mode == "rrg":
                    # 0=not_reported, 1=present, 2=absent, 3=uncertain
                    # rrg mode: present or uncertain -> 1, else 0
                    labels.append([1 if c in {1, 3} else 0 for c in row])
                elif mode == "classification":
                    labels.append([1 if c == 1 else -1 if c == 3 else 0 for c in row])
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        return labels

    def forward(self, hyps: List[str], refs: List[str], on_batch_done=None):
        refs_labels = self.get_labels(refs)
        hyps_labels = self.get_labels(hyps)

        refs5 = [np.array(r)[self.top5_idx] for r in refs_labels]
        hyps5 = [np.array(h)[self.top5_idx] for h in hyps_labels]

        y_true5 = np.asarray(refs5)
        y_pred5 = np.asarray(hyps5)
        sample_label_acc_5 = (y_true5 == y_pred5).mean(axis=1).astype(float).tolist()

        # Aggregate sample accuracy only over reliable findings (test F1 >= 0.3)
        y_true_full = np.asarray(refs_labels)[:, RELIABLE_IDX]
        y_pred_full = np.asarray(hyps_labels)[:, RELIABLE_IDX]
        sample_label_acc_full = (y_true_full == y_pred_full).mean(axis=1).astype(float).tolist()

        refs_reliable = [[r[i] for i in RELIABLE_IDX] for r in refs_labels]
        hyps_reliable = [[h[i] for i in RELIABLE_IDX] for h in hyps_labels]
        cr_all = classification_report(
            refs_reliable, hyps_reliable, target_names=RELIABLE_FINDINGS, output_dict=True)
        cr_5 = classification_report(
            refs5, hyps5, target_names=TOP5, output_dict=True)

        accuracy = accuracy_score(refs5, hyps5)
        pe_accuracy = (np.count_nonzero(y_true5 - y_pred5, axis=1) == 0).astype(float)

        return accuracy, pe_accuracy, cr_all, cr_5, sample_label_acc_full, sample_label_acc_5
