"""HopprF1CthRrgLv001ClsLv003: multi-output CT-head report evaluator (ModernBERT-large).

Convention-renamed copy of f1hopprchexbert_cth, following the input-based metric
naming f1hoppr_<rrg_version>_<cls_version>. Same model and 3-class logic;
trained on cth_rrg_internal_lv001 (clean_findings) + cth_cls_internal_lv003.

A single forward pass classifies 10 CT-head incidental/chronic findings. Each
finding head outputs 3-way logits (0 = absent or not reported, 1 = present,
2 = uncertain), collapsed to binary for F1: positive = {present, uncertain}
(predicted class != 0), negative = absent.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

_DEFAULT_CKPT = (
    "/nfs/cluster/hoppr_vlm_ressources/radeval_checkpoints/f1hoppr_cth_rrg_lv001_cls_lv003"
)

CONDITION_NAMES = OrderedDict([
    ("atrophy", "Atrophy"),
    ("white_matter_disease", "White matter disease"),
    ("sinus_disease", "Sinus disease"),
    ("post_surgical_calvarium", "Post-surgical calvarium"),
    ("intracranial_atherosclerosis", "Intracranial atherosclerosis"),
    ("lens_replacement", "Lens replacement"),
    ("scleral_buckle", "Scleral buckle"),
    ("technical_limitation", "Technical limitation"),
    ("encephalomalacia", "Encephalomalacia"),
    ("mastoid_effusion", "Mastoid effusion"),
])

NUM_CONDITIONS = len(CONDITION_NAMES)
NUM_CLASSES = 3  # 0 = absent or not reported, 1 = present, 2 = uncertain

# ---------------------------------------------------------------------------
# Model definition (must match the training code for from_pretrained to work)
# ---------------------------------------------------------------------------


@dataclass
class MultiOutputClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class MultiOutputClassifier(PreTrainedModel):
    """10 independent 3-class heads sharing one BERT-style encoder."""

    config_class = AutoConfig
    _keys_to_ignore_on_load_unexpected = [r"cls", r"classifier", r"score"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        hidden = config.hidden_size
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, NUM_CLASSES) for _ in range(NUM_CONDITIONS)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MultiOutputClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]
        all_logits = torch.stack([head(cls_hidden) for head in self.heads], dim=1)

        loss = None
        if labels is not None:
            total_loss = torch.tensor(0.0, device=all_logits.device,
                                      dtype=all_logits.dtype)
            for i in range(NUM_CONDITIONS):
                total_loss = total_loss + nn.functional.cross_entropy(
                    all_logits[:, i, :], labels[:, i])
            loss = total_loss / NUM_CONDITIONS

        return MultiOutputClassifierOutput(loss=loss, logits=all_logits)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HopprF1CthRrgLv001ClsLv003:
    """Multi-output CT-head finding classifier for report evaluation.

    A single forward pass produces (batch, 10, 3) logits. Predictions are
    collapsed to binary (class 0 = negative; classes 1, 2 = positive, i.e.
    present or uncertain) and compared via sklearn classification_report.
    """

    LABELS = list(CONDITION_NAMES.keys())
    NO_FINDING = "no_finding"

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CKPT,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 16,
        max_length: int = 1024,
    ):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"HopprF1CthRrgLv001ClsLv003 checkpoint not found: {checkpoint_dir}")

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token)

        self.model = MultiOutputClassifier.from_pretrained(
            checkpoint_dir, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict_label_matrix(
        self, reports: Sequence[str], on_batch_done=None,
    ) -> np.ndarray:
        """Return binary label matrix of shape (N, 11).

        Logits (N, 10, 3) -> argmax -> binary (classes 1, 2 = positive).
        An 11th "no_finding" column is 1 when all 10 findings are 0.
        """
        all_binary = []
        report_list = list(reports)

        for start in range(0, len(report_list), self.batch_size):
            batch = report_list[start:start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # (B, 10, 3)
            pred_ids = logits.argmax(dim=-1)   # (B, 10)
            binary = (pred_ids != 0).int().cpu()  # present or uncertain -> positive
            all_binary.append(binary)
            if on_batch_done:
                on_batch_done()

        matrix = torch.cat(all_binary, dim=0)  # (N, 10)
        no_finding = (~matrix.any(dim=1)).unsqueeze(1).int()
        full = torch.cat([matrix, no_finding], dim=1)
        return full.numpy()

    def __call__(self, hyps: List[str], refs: List[str], on_batch_done=None):
        return self.forward(hyps=hyps, refs=refs, on_batch_done=on_batch_done)

    def forward(
        self, hyps: List[str], refs: List[str], on_batch_done=None,
    ) -> Tuple[float, List[float], dict]:
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if len(hyps) == 0:
            return 0.0, [], {}

        y_pred = self._predict_label_matrix(hyps, on_batch_done=on_batch_done)
        y_true = self._predict_label_matrix(refs, on_batch_done=on_batch_done)

        accuracy = float(accuracy_score(y_true, y_pred))
        per_sample_accuracy = (y_true == y_pred).all(axis=1).astype(float).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=self.LABELS + [self.NO_FINDING],
            output_dict=True,
            zero_division=0,
        )
        return accuracy, per_sample_accuracy, report
