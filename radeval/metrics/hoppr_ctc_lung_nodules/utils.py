"""Utility helpers for the hoppr_ctc_lung_nodules metric.

Responsibilities:
    - `extract_pn_segment(clean_findings)`: pull the PULMONARY NODULES: segment from
      a full clean_findings string. Returns "" if not present.
    - `extract_json_str(raw)`: strip markdown fences / repair truncated JSON from
      an LLM response.
    - `compute_per_row_metrics(parsed)`: deterministic scoring from the LLM's
      validated JSON output. Emits all per-row scores the metric exposes,
      including per-bucket detection counts (TP/FN/FP) keyed by ref-side bucket
      for sensitivity/FP-per-study aggregation, per-bucket diameter errors for
      MAE/MAPE aggregation, and density (solid vs sub-solid) / calcified /
      laterality / lobe correctness counts for accuracy aggregation.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional


SIZE_BUCKETS: tuple[str, ...] = ("lt4", "4to6", "6to8", "8to15", "gt15")


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

# All 8 section headers used by lv004/lv005 clean_findings format.
# Order matters for the "next header" lookup regex: any of these terminates
# the PULMONARY NODULES: segment.
_ALL_HEADERS = [
    "LUNGS AND AIRWAYS",
    "PULMONARY NODULES",
    "MEDIASTINUM",
    "HEART AND GREAT VESSELS",
    "UPPER ABDOMEN",
    "BONES",
    "SUPPORT DEVICES",
    "SOFT TISSUES",
]
_OTHER_HEADERS = [h for h in _ALL_HEADERS if h != "PULMONARY NODULES"]

# Matches `PULMONARY NODULES: <content>` up to the next top-level header or EOS.
# `\s` covers the space / newline before the next header.
_PN_SEGMENT_RE = re.compile(
    r"PULMONARY NODULES:\s*(.*?)"
    r"(?=\s(?:" + "|".join(re.escape(h) for h in _OTHER_HEADERS) + r"):|$)",
    re.DOTALL,
)


def extract_pn_segment(clean_findings: str) -> str:
    """Extract the content of the PULMONARY NODULES: section.

    Returns the section body (without the `PULMONARY NODULES: ` prefix) or
    an empty string if the section is absent.
    """
    if not clean_findings or not isinstance(clean_findings, str):
        return ""
    m = _PN_SEGMENT_RE.search(clean_findings)
    if not m:
        return ""
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Canonical sentence ordering
# ---------------------------------------------------------------------------
#
# The LLM judge is mostly order-invariant on clean inputs but can make
# different one-to-one matching choices when the predicted nodule list is
# reordered (observed empirically — see tests). We normalize both reference
# and predicted nodule sentences into a canonical order before prompting so
# the judge sees the same input regardless of how the source text wrote it.
# Order key: (laterality, lobe, size, raw text). Sentences that don't match
# the simple `There is/are ... nodule|mass in the ...` shape fall through to
# a stable trailing block in original order.

_LATERALITY_ORDER: dict[str, int] = {
    "right": 0,
    "left": 1,
    "bilateral": 2,
    "unspecified": 3,
}

_LOBE_ORDER: list[tuple[str, str]] = [
    # (substring to match, lobe key)
    ("right upper lobe", "right upper lobe"),
    ("right middle lobe", "right middle lobe"),
    ("right lower lobe", "right lower lobe"),
    ("right lung",       "right lung"),
    ("left upper lobe",  "left upper lobe"),
    ("lingula",          "lingula"),
    ("left lower lobe",  "left lower lobe"),
    ("left lung",        "left lung"),
    ("bilateral lungs",  "bilateral lungs"),
    ("lung",             "lung"),
]
_LOBE_RANK: dict[str, int] = {key: i for i, (_, key) in enumerate(_LOBE_ORDER)}
_LOBE_RANK["unspecified"] = len(_LOBE_RANK)


def _laterality_from_lobe(lobe: str) -> str:
    if lobe in ("right upper lobe", "right middle lobe",
                "right lower lobe", "right lung"):
        return "right"
    if lobe in ("left upper lobe", "lingula",
                "left lower lobe", "left lung"):
        return "left"
    if lobe == "bilateral lungs":
        return "bilateral"
    return "unspecified"


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.IGNORECASE)


def _largest_size_mm(sentence: str) -> Optional[float]:
    """Largest numeric size mentioned in the sentence, in mm. None if absent.

    For multi-dim phrases like "1.6 x 13 mm", returns the largest dimension
    expressed in the same unit. cm values are converted to mm.
    """
    best: Optional[float] = None
    for m in _SIZE_RE.finditer(sentence):
        value = float(m.group(1))
        if m.group(2).lower() == "cm":
            value *= 10.0
        if best is None or value > best:
            best = value
    return best


_NODULE_SHAPE_RE = re.compile(r"\bnodul|\bmass\b", re.IGNORECASE)


def _split_sentences(segment: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace, keeping
    the punctuation. Returns a list of trimmed sentences with a trailing
    period restored.
    """
    if not segment:
        return []
    parts = re.split(r"(?<=[.!?])\s+", segment.strip())
    cleaned: list[str] = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if not s.endswith((".", "!", "?")):
            s = s + "."
        cleaned.append(s)
    return cleaned


def _sort_key(sentence: str, idx: int) -> tuple:
    """Build a deterministic sort key for a single nodule sentence.

    The key is (is_other, laterality_rank, lobe_rank, size_or_inf,
    sentence_lower, idx). `is_other` is 1 for sentences that don't look
    like a nodule sentence (e.g. summary statements) so they trail
    naturally without disrupting the canonical order.
    """
    s_low = sentence.lower()
    is_other = 0 if _NODULE_SHAPE_RE.search(s_low) else 1

    lobe = "unspecified"
    for substr, key in _LOBE_ORDER:
        if substr in s_low:
            lobe = key
            break
    laterality = _laterality_from_lobe(lobe)
    size_mm = _largest_size_mm(sentence)
    size_or_inf = size_mm if size_mm is not None else float("inf")
    return (
        is_other,
        _LATERALITY_ORDER[laterality],
        _LOBE_RANK[lobe],
        size_or_inf,
        s_low,
        idx,
    )


def canonicalize_pn_segment(segment: str) -> str:
    """Sort nodule sentences into a canonical order.

    Order: laterality (right < left < bilateral < unspecified), then lobe
    (RUL < RML < RLL < right lung < LUL < lingula < LLL < left lung <
    bilateral lungs < lung), then size ascending in mm (None last), then
    case-folded sentence text, then original index. Sentences that do not
    look like nodule statements (no `nodule` / `mass`) trail in original
    order.

    Returns the canonicalized segment as a single space-joined string.
    Empty input -> "".
    """
    sentences = _split_sentences(segment)
    if not sentences:
        return ""
    indexed = list(enumerate(sentences))
    indexed.sort(key=lambda pair: _sort_key(pair[1], pair[0]))
    return " ".join(s for _, s in indexed)



# ---------------------------------------------------------------------------
# JSON cleanup / validation
# ---------------------------------------------------------------------------

def extract_json_str(text: str) -> str:
    """Strip markdown fences and trailing commas from an LLM JSON response."""
    t = text.strip()
    md = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.DOTALL)
    if md:
        t = md.group(1).strip()
    else:
        start = t.find("{")
        end_brace = t.rfind("}")
        if start >= 0 and end_brace > start:
            t = t[start:end_brace + 1]
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


def parse_json_response(raw: str) -> dict:
    """Parse a JSON response. Raises ValueError on persistent failure."""
    cleaned = extract_json_str(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse hoppr_ctc_lung_nodules JSON: {e}. Raw: {raw[:400]}"
        )


def validate_response(data: dict) -> None:
    """Validate the LLM response has the expected structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    for key in (
        "reference_nodules",
        "predicted_nodules",
        "matched_pairs",
        "false_findings",
        "missing_findings",
    ):
        if key not in data:
            raise ValueError(
                f"Missing required key '{key}' in hoppr_ctc_lung_nodules response"
            )


# ---------------------------------------------------------------------------
# Per-row scoring (deterministic, pure-Python from parsed JSON)
# ---------------------------------------------------------------------------

# CRIMSON-inspired attribute severity weight for the composite score.
_ATTR_ERROR_WEIGHT = 0.5

# Per-nodule base weight — every nodule counts equally in this metric
# (no CRIMSON-style significance weighting; the LLM doesn't classify them).
_NODULE_BASE_WEIGHT = 1.0

# Attribute errors that reduce the matched-pair credit in the composite
# score. `noun_error` (nodule vs mass) is intentionally excluded; the
# consolidated VLM desiderata do not require it.
_COMPOSITE_ATTR_ERROR_KEYS: tuple[str, ...] = (
    "size_error",
    "density_error",
    "calcified_error",
    "location_error",
    "uncertainty_error",
)


def _is_numeric(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _bucket_from_size(size_mm) -> Optional[str]:
    """Return the canonical bucket name for a numeric size in mm, or None.

    Used as a fallback when the LLM omits `size_bucket` for a nodule that
    has a numeric `size_mm`.
    """
    if not _is_numeric(size_mm):
        return None
    s = float(size_mm)
    if s < 4:
        return "lt4"
    if s < 6:
        return "4to6"
    if s <= 8:
        return "6to8"
    if s <= 15:
        return "8to15"
    return "gt15"


def _norm_bucket(nodule: dict) -> Optional[str]:
    """Pick a canonical bucket from a nodule dict.

    Prefers the LLM-emitted `size_bucket`; falls back to a Python-side
    derivation from `size_mm` so we never lose a TP because the LLM
    forgot the bucket.
    """
    bucket = nodule.get("size_bucket")
    if bucket in SIZE_BUCKETS:
        return bucket
    return _bucket_from_size(nodule.get("size_mm"))


def _norm_density(d) -> Optional[str]:
    """Normalize a density string for downstream comparison."""
    if not isinstance(d, str):
        return None
    d = d.strip().lower()
    if d in {"solid", "part-solid", "ground-glass"}:
        return d
    if d in {"part solid"}:
        return "part-solid"
    if d in {"groundglass", "ground glass"}:
        return "ground-glass"
    return None


def _is_solid(d: Optional[str]) -> Optional[bool]:
    """Map normalized density to two-class solid/sub-solid (True/False) or None."""
    if d == "solid":
        return True
    if d in {"part-solid", "ground-glass"}:
        return False
    return None


def _zero_bucket_dict() -> dict[str, int]:
    return {b: 0 for b in SIZE_BUCKETS}


def compute_per_row_metrics(parsed: dict) -> dict[str, Any]:
    """Deterministic per-row scoring from the parsed LLM JSON.

    Returns a dict with all per-row metric values plus error counts and
    per-bucket counters. Never raises (any ambiguity resolves to NaN or 0
    depending on the field).

    Key behaviors:
    - Detection P/R/F1 derived from counts of matched_pairs vs false/missing.
    - Per-bucket detection counters keyed by ref-side bucket for TP/FN and
      by pred-side bucket for FP. These feed the desiderata-level
      sensitivity / FP-per-study aggregations.
    - Size accuracy / exact_match / MAE / MAPE computed ONLY over matched
      pairs where BOTH sides have a numeric size; per-bucket variants use
      the ref-side bucket.
    - Density two-class accuracy (solid vs sub-solid) computed over matched
      pairs where BOTH sides have a non-null density.
    - Calcified accuracy computed over matched pairs where BOTH sides have
      a non-null calcified flag.
    - Laterality / lobe accuracy computed over all matched pairs (with
      laterality / location available on both sides).
    - Composite follows CRIMSON's S-shaped formula:
        S = (correct - penalty) / N_G
        correct = sum over matched pairs of base_weight * credit_factor
        penalty = sum of false-finding weights
        N_G     = sum over reference nodules of base_weight
      crimson_score = S if S >= 0 else -d/(1+d), d = penalty - correct
      Attribute errors counted in the credit factor: size, density,
      calcified, location, uncertainty (noun_error excluded by design).
    """
    ref_list = parsed.get("reference_nodules") or []
    pred_list = parsed.get("predicted_nodules") or []
    matched = parsed.get("matched_pairs") or []
    false_findings = parsed.get("false_findings") or []
    missing_findings = parsed.get("missing_findings") or []

    n_ref = len(ref_list)
    n_pred = len(pred_list)
    n_matched = len(matched)
    n_false = len(false_findings)
    n_miss = len(missing_findings)

    # Build id -> nodule lookup tables for per-bucket counting.
    ref_by_id = {n.get("id"): n for n in ref_list if n.get("id")}
    pred_by_id = {n.get("id"): n for n in pred_list if n.get("id")}

    # ------------------------------------------------------------------
    # Detection P/R/F1 (corpus-aggregable; null when undefined)
    # ------------------------------------------------------------------
    precision = None
    recall = None
    f1 = None
    if n_pred > 0:
        precision = n_matched / (n_matched + n_false) if (n_matched + n_false) > 0 else 0.0
    if n_ref > 0:
        recall = n_matched / (n_matched + n_miss) if (n_matched + n_miss) > 0 else 0.0
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0

    # ------------------------------------------------------------------
    # Per-bucket detection counters (TP/FN by ref bucket, FP by pred bucket)
    # ------------------------------------------------------------------
    tp_by_bucket = _zero_bucket_dict()
    fn_by_bucket = _zero_bucket_dict()
    fp_by_bucket = _zero_bucket_dict()

    matched_ref_ids: set[str] = set()
    for m in matched:
        rid = m.get("ref_id")
        if rid:
            matched_ref_ids.add(rid)
        ref_bucket = m.get("ref_size_bucket")
        if ref_bucket not in SIZE_BUCKETS and rid in ref_by_id:
            ref_bucket = _norm_bucket(ref_by_id[rid])
        if ref_bucket in SIZE_BUCKETS:
            tp_by_bucket[ref_bucket] += 1

    for rid in missing_findings:
        rb = _norm_bucket(ref_by_id.get(rid, {})) if rid in ref_by_id else None
        if rb in SIZE_BUCKETS:
            fn_by_bucket[rb] += 1

    for pid in false_findings:
        pb = _norm_bucket(pred_by_id.get(pid, {})) if pid in pred_by_id else None
        if pb in SIZE_BUCKETS:
            fp_by_bucket[pb] += 1

    # ------------------------------------------------------------------
    # Size metrics (over matched pairs with both numeric sizes)
    # ------------------------------------------------------------------
    size_acc_hits = size_exact_hits = 0
    size_samples = 0
    abs_errors_mm: list[float] = []
    pct_errors: list[float] = []
    abs_errors_by_bucket: dict[str, list[float]] = {b: [] for b in SIZE_BUCKETS}
    pct_errors_by_bucket: dict[str, list[float]] = {b: [] for b in SIZE_BUCKETS}

    for m in matched:
        ref_sz = m.get("ref_size_mm")
        pred_sz = m.get("pred_size_mm")
        if not (_is_numeric(ref_sz) and _is_numeric(pred_sz)):
            continue
        size_samples += 1
        if not m.get("size_error", False):
            size_acc_hits += 1
        exact_flag = m.get("size_exact_match")
        if exact_flag is None:
            exact_flag = (float(ref_sz) == float(pred_sz))
        if exact_flag:
            size_exact_hits += 1
        err = abs(float(pred_sz) - float(ref_sz))
        abs_errors_mm.append(err)
        if float(ref_sz) > 0:
            pct_errors.append(err / float(ref_sz))
        ref_bucket = m.get("ref_size_bucket")
        if ref_bucket not in SIZE_BUCKETS:
            ref_bucket = _bucket_from_size(ref_sz)
        if ref_bucket in SIZE_BUCKETS:
            abs_errors_by_bucket[ref_bucket].append(err)
            if float(ref_sz) > 0:
                pct_errors_by_bucket[ref_bucket].append(err / float(ref_sz))

    size_accuracy = (size_acc_hits / size_samples) if size_samples else None
    size_exact_match = (size_exact_hits / size_samples) if size_samples else None
    size_mae_mm = (sum(abs_errors_mm) / len(abs_errors_mm)) if abs_errors_mm else None
    size_mape = (sum(pct_errors) / len(pct_errors)) if pct_errors else None

    # ------------------------------------------------------------------
    # Density two-class accuracy (solid vs sub-solid), calcified, locations
    # ------------------------------------------------------------------
    density_correct_solid = density_total_solid = 0
    density_correct_subsolid = density_total_subsolid = 0
    calcified_correct = calcified_total = 0
    laterality_correct = laterality_total = 0
    lobe_correct = lobe_total = 0

    for m in matched:
        rid = m.get("ref_id")
        pid = m.get("pred_id")
        ref_n = ref_by_id.get(rid, {}) if rid else {}
        pred_n = pred_by_id.get(pid, {}) if pid else {}

        r_dens = _norm_density(ref_n.get("density"))
        p_dens = _norm_density(pred_n.get("density"))
        r_solid = _is_solid(r_dens)
        p_solid = _is_solid(p_dens)
        if r_solid is not None and p_solid is not None:
            if r_solid is True:
                density_total_solid += 1
                if p_solid is True:
                    density_correct_solid += 1
            else:
                density_total_subsolid += 1
                if p_solid is False:
                    density_correct_subsolid += 1

        r_calc = ref_n.get("calcified")
        p_calc = pred_n.get("calcified")
        if isinstance(r_calc, bool) and isinstance(p_calc, bool):
            calcified_total += 1
            if r_calc == p_calc:
                calcified_correct += 1

        r_lat = ref_n.get("laterality")
        p_lat = pred_n.get("laterality")
        if r_lat and p_lat:
            laterality_total += 1
            if r_lat == p_lat:
                laterality_correct += 1

        r_loc = ref_n.get("location")
        p_loc = pred_n.get("location")
        if r_loc and p_loc:
            lobe_total += 1
            if not m.get("location_error", False):
                lobe_correct += 1

    density_accuracy_solid = (
        density_correct_solid / density_total_solid
        if density_total_solid else None
    )
    density_accuracy_subsolid = (
        density_correct_subsolid / density_total_subsolid
        if density_total_subsolid else None
    )
    if density_total_solid + density_total_subsolid > 0:
        density_accuracy_micro = (
            (density_correct_solid + density_correct_subsolid)
            / (density_total_solid + density_total_subsolid)
        )
    else:
        density_accuracy_micro = None

    calcified_accuracy = (
        calcified_correct / calcified_total if calcified_total else None
    )
    laterality_accuracy = (
        laterality_correct / laterality_total if laterality_total else None
    )
    location_accuracy = (lobe_correct / lobe_total) if lobe_total else None

    # Diagnostic accuracies (noun, uncertainty) over all matched pairs.
    def _acc(field_err: str) -> Optional[float]:
        if not matched:
            return None
        hits = sum(1 for m in matched if not m.get(field_err, False))
        return hits / len(matched)

    noun_accuracy = _acc("noun_error")
    uncertainty_accuracy = _acc("uncertainty_error")

    # ------------------------------------------------------------------
    # CRIMSON-style composite (noun_error excluded from credit factor)
    # ------------------------------------------------------------------
    N_G = n_ref * _NODULE_BASE_WEIGHT
    penalty = n_false * _NODULE_BASE_WEIGHT

    correct = 0.0
    for m in matched:
        base = _NODULE_BASE_WEIGHT
        n_attr_errors = sum(
            1 for k in _COMPOSITE_ATTR_ERROR_KEYS
            if m.get(k, False)
        )
        sum_attr_weights = n_attr_errors * _ATTR_ERROR_WEIGHT
        denom = base + sum_attr_weights
        credit_factor = base / denom if denom > 0 else 0.0
        correct += base * credit_factor

    if N_G == 0 and n_pred == 0:
        composite = 1.0
        S = 1.0
    elif N_G == 0:
        S = -(penalty + 1)
        composite = -penalty / (1 + penalty) if penalty > 0 else 0.0
    else:
        S = (correct - penalty) / N_G
        if S >= 0:
            composite = S
        else:
            d = penalty - correct
            composite = -d / (1 + d) if d > 0 else 0.0

    # ------------------------------------------------------------------
    # Per-bucket size error vectors as flat columns (one per bucket).
    # ------------------------------------------------------------------
    sum_abs_by_bucket = {b: float(sum(abs_errors_by_bucket[b])) for b in SIZE_BUCKETS}
    n_abs_by_bucket = {b: len(abs_errors_by_bucket[b]) for b in SIZE_BUCKETS}
    sum_pct_by_bucket = {b: float(sum(pct_errors_by_bucket[b])) for b in SIZE_BUCKETS}
    n_pct_by_bucket = {b: len(pct_errors_by_bucket[b]) for b in SIZE_BUCKETS}

    out: dict[str, Any] = {
        # Detection (per-row)
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        # Size (per-row)
        "size_accuracy": size_accuracy,
        "size_exact_match": size_exact_match,
        "size_mae_mm": size_mae_mm,
        "size_mape": size_mape,
        "n_size_pairs": size_samples,
        # Density / calcified / location (per-row)
        "density_accuracy_solid": density_accuracy_solid,
        "density_accuracy_subsolid": density_accuracy_subsolid,
        "density_accuracy_micro": density_accuracy_micro,
        "calcified_accuracy": calcified_accuracy,
        "laterality_accuracy": laterality_accuracy,
        "location_accuracy": location_accuracy,
        "noun_accuracy": noun_accuracy,
        "uncertainty_accuracy": uncertainty_accuracy,
        # Legacy alias for backward-compat consumers; same value as
        # density_accuracy_micro (two-class density). Kept so existing
        # downstream code keying on `type_accuracy` still works.
        "type_accuracy": density_accuracy_micro,
        # Composite (per-row)
        "composite": composite,
        # Raw counts (per-row)
        "n_reference": n_ref,
        "n_predicted": n_pred,
        "n_matched": n_matched,
        "n_false_findings": n_false,
        "n_missing_findings": n_miss,
        "n_size_errors": sum(1 for m in matched if m.get("size_error")),
        "n_density_errors": sum(1 for m in matched if m.get("density_error")),
        "n_calcified_errors": sum(1 for m in matched if m.get("calcified_error")),
        "n_location_errors": sum(1 for m in matched if m.get("location_error")),
        "n_noun_errors": sum(1 for m in matched if m.get("noun_error")),
        "n_uncertainty_errors": sum(1 for m in matched if m.get("uncertainty_error")),
        # Density / calcified / loc raw counts
        "density_correct_solid": density_correct_solid,
        "density_total_solid": density_total_solid,
        "density_correct_subsolid": density_correct_subsolid,
        "density_total_subsolid": density_total_subsolid,
        "calcified_correct": calcified_correct,
        "calcified_total": calcified_total,
        "laterality_correct": laterality_correct,
        "laterality_total": laterality_total,
        "lobe_correct": lobe_correct,
        "lobe_total": lobe_total,
    }
    # Per-bucket detection counters as flat columns.
    for b in SIZE_BUCKETS:
        out[f"tp_{b}"] = tp_by_bucket[b]
        out[f"fn_{b}"] = fn_by_bucket[b]
        out[f"fp_{b}"] = fp_by_bucket[b]
        out[f"abs_err_sum_{b}"] = sum_abs_by_bucket[b]
        out[f"abs_err_n_{b}"] = n_abs_by_bucket[b]
        out[f"pct_err_sum_{b}"] = sum_pct_by_bucket[b]
        out[f"pct_err_n_{b}"] = n_pct_by_bucket[b]
    return out


def empty_row_result(both_empty: bool = True) -> dict[str, Any]:
    """Per-row result for a row where neither ref nor hyp has a PN section.

    Both sides empty -> composite=1.0 (perfect "agreement on no nodules"),
    all rate-style metric fields None so they don't pollute aggregates.
    Per-bucket counters are zeroed (so the corpus-level FP/study correctly
    treats this study as having 0 FPs in every bucket).
    """
    composite = 1.0 if both_empty else None
    out: dict[str, Any] = {
        "detection_precision": None,
        "detection_recall": None,
        "detection_f1": None,
        "size_accuracy": None,
        "size_exact_match": None,
        "size_mae_mm": None,
        "size_mape": None,
        "n_size_pairs": 0,
        "density_accuracy_solid": None,
        "density_accuracy_subsolid": None,
        "density_accuracy_micro": None,
        "calcified_accuracy": None,
        "laterality_accuracy": None,
        "location_accuracy": None,
        "noun_accuracy": None,
        "uncertainty_accuracy": None,
        "type_accuracy": None,
        "composite": composite,
        "n_reference": 0,
        "n_predicted": 0,
        "n_matched": 0,
        "n_false_findings": 0,
        "n_missing_findings": 0,
        "n_size_errors": 0,
        "n_density_errors": 0,
        "n_calcified_errors": 0,
        "n_location_errors": 0,
        "n_noun_errors": 0,
        "n_uncertainty_errors": 0,
        "density_correct_solid": 0,
        "density_total_solid": 0,
        "density_correct_subsolid": 0,
        "density_total_subsolid": 0,
        "calcified_correct": 0,
        "calcified_total": 0,
        "laterality_correct": 0,
        "laterality_total": 0,
        "lobe_correct": 0,
        "lobe_total": 0,
    }
    for b in SIZE_BUCKETS:
        out[f"tp_{b}"] = 0
        out[f"fn_{b}"] = 0
        out[f"fp_{b}"] = 0
        out[f"abs_err_sum_{b}"] = 0.0
        out[f"abs_err_n_{b}"] = 0
        out[f"pct_err_sum_{b}"] = 0.0
        out[f"pct_err_n_{b}"] = 0
    return out



