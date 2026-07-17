"""Shared scorer for private 3-class multi-output f1hoppr_* metrics."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class MultiOutputClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class MultiOutputClassifier(PreTrainedModel):
    """Dynamic N-head, 3-class model matching text_classifiers.model."""

    config_class = AutoConfig
    _keys_to_ignore_on_load_unexpected = [r"cls", r"classifier", r"score"]

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        self.num_conditions = int(config.num_conditions)
        self.num_classes = int(config.num_classes)
        self.condition_names = list(config.condition_names)
        self.pooling = getattr(config, "text_classifier_pooling", "cls")
        self.heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, self.num_classes)
                for _ in range(self.num_conditions)
            ]
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
        hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            pooled = hidden[:, 0]
        logits = torch.stack([head(pooled) for head in self.heads], dim=1)
        loss = None
        if labels is not None:
            total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            for idx in range(self.num_conditions):
                total = total + nn.functional.cross_entropy(
                    logits[:, idx, :], labels[:, idx]
                )
            loss = total / self.num_conditions
        return MultiOutputClassifierOutput(loss=loss, logits=logits)


class HopprMultiOutputScorer:
    """Compare generated/reference reports through a 3-class finding model."""

    NO_FINDING = "no_finding"

    def __init__(
        self,
        checkpoint_dir: str,
        expected_conditions: Sequence[str],
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"f1hoppr checkpoint not found: {checkpoint_dir}")
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        self.model = MultiOutputClassifier.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.LABELS = list(self.model.condition_names)
        expected = list(expected_conditions)
        if self.LABELS != expected:
            raise ValueError(
                "Checkpoint condition order does not match metric definition: "
                f"checkpoint={self.LABELS}, expected={expected}"
            )
        if self.model.num_classes != 3:
            raise ValueError(
                f"Expected 3 classes (absent/present/uncertain), got "
                f"{self.model.num_classes}"
            )

    @torch.no_grad()
    def _predict_label_matrix(
        self, reports: Sequence[str], on_batch_done=None
    ) -> np.ndarray:
        all_binary = []
        report_list = list(reports)
        for start in range(0, len(report_list), self.batch_size):
            batch = report_list[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(self.device) for key, value in enc.items()}
            pred_ids = self.model(**enc).logits.argmax(dim=-1)
            all_binary.append((pred_ids != 0).int().cpu())
            if on_batch_done:
                on_batch_done()
        matrix = torch.cat(all_binary, dim=0)
        no_finding = (~matrix.any(dim=1)).unsqueeze(1).int()
        return torch.cat([matrix, no_finding], dim=1).numpy()

    def __call__(self, hyps: List[str], refs: List[str], on_batch_done=None):
        if not isinstance(hyps, list) or not isinstance(refs, list):
            raise TypeError("hyps and refs must be of type list")
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs lists don't have the same size")
        if not hyps:
            return 0.0, [], {}
        y_pred = self._predict_label_matrix(hyps, on_batch_done=on_batch_done)
        y_true = self._predict_label_matrix(refs, on_batch_done=on_batch_done)
        accuracy = float(accuracy_score(y_true, y_pred))
        sample_acc = (y_true == y_pred).all(axis=1).astype(float).tolist()
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.LABELS + [self.NO_FINDING],
            output_dict=True,
            zero_division=0,
        )
        return accuracy, sample_acc, report
