"""Prompt parts for the hoppr_ctc_lung_nodules LLM-as-judge metric.

Compact, composable sections following the CRIMSON / MammoGREEN pattern.
`build_prompt(ref_segment, hyp_segment)` assembles the full user prompt.

Design notes:
    - The input being judged is ONLY the `PULMONARY NODULES:` subsection of a
      chest CT findings report. The system message and the user prompt say so
      explicitly so the LLM does not invent other anatomy.
    - The schema separates `density` (solid / part-solid / ground-glass) from
      `calcified` so density accuracy can be reported as solid vs sub-solid
      (head-of-ML desideratum), with calcified tracked as a separate benign
      attribute.
    - Each reference and predicted nodule carries a `size_bucket`
      (lt4 / 4to6 / 6to8 / 8to15 / gt15 / null) so Python can aggregate
      per-bucket sensitivity, FP-per-study, and diameter MAE deterministically.
    - Size tolerances mirror the consolidated diameter MAE targets:
        4-6 mm   : +/- 1.0 mm
        6-8 mm   : +/- 1.25 mm
        > 8 mm   : +/- 15% relative
    - All scoring is deterministic in Python from the JSON output. The LLM
      only extracts structure and flags attribute errors.
"""

SYSTEM_MSG = (
    "You are a radiology AI evaluator specializing in pulmonary nodule "
    "comparison. The ONLY input you will see is the content of the "
    "`PULMONARY NODULES:` subsection of a chest CT findings report — first "
    "for the reference, then for the prediction. Do NOT consider lungs, "
    "airways, mediastinum, or any other anatomy. You compare predicted "
    "nodule descriptions against the reference, identify matches, misses, "
    "false findings, and attribute errors. Always respond with strictly "
    "valid JSON only."
)


OBJECTIVE = """\
Objective:

You are evaluating the `PULMONARY NODULES:` subsection of a chest CT findings
report. Both REFERENCE NODULES and PREDICTED NODULES below contain ONLY that
subsection — no other anatomy.

Compare the PREDICTED list of pulmonary nodules / masses to the REFERENCE
list. For each side, parse every sentence into a structured nodule object,
then determine matches, false findings, and attribute errors. This is the
sole evaluation of positive pulmonary focal lesions; you do NOT consider
any other anatomy. Negated nodule statements (e.g. "No pulmonary nodules.")
do not appear in these inputs; ignore them if seen."""


TEMPLATE_REF = """\
Input format:

Each side is a sequence of sentences produced by a templated generator. The
target template is:

    There is a [size] [density] [calcified] nodule in the [location].
    There is a [size] [density] [calcified] mass in the [location].
    There are multiple [size] [density] nodules in the [location].
    There is possibly a [size] [density] nodule in the [location].  (existence uncertain)

Any slot may be omitted if not stated. Examples:
    "There is an 8 mm solid nodule in the right upper lobe."
    "There is a 4 mm nodule in the left lower lobe."  (no density, no calcified)
    "There is a solid nodule in the right middle lobe." (no size)
    "There is a 3.2 cm mass in the right lower lobe."
    "There are multiple calcified nodules in the bilateral lungs."
    "There is possibly a 5 mm ground-glass nodule in the lingula."
    "There is a 6 mm calcified nodule in the right upper lobe."
    "There is a 7 mm part-solid nodule in the left upper lobe."

The predicted side may deviate from the template — parse it robustly."""


NODULE_FIELDS = """\
Nodule fields to extract per sentence:
    - id:         "R1", "R2", ... for reference; "P1", "P2", ... for predicted.
    - size_mm:    numeric in millimeters if stated; otherwise null.
                  Convert cm -> mm (e.g. "2.3 cm" -> 23). For ranges
                  ("5-7 mm"), use the LARGER value. For dimensions
                  ("15 x 12 mm"), use the largest dimension.
    - size_bucket: derived from size_mm. Use exactly one of:
                  "lt4"   if size_mm < 4,
                  "4to6"  if 4 <= size_mm < 6,
                  "6to8"  if 6 <= size_mm <= 8,
                  "8to15" if 8 < size_mm <= 15,
                  "gt15"  if size_mm > 15,
                  null    if size_mm is null.
    - density:    one of "solid", "part-solid", "ground-glass"; otherwise null.
                  Treat unspecified solid-appearing nodules (most cases) as
                  null, NOT "solid", unless the source explicitly says "solid".
                  Normalize "groundglass"/"ground glass" -> "ground-glass".
                  Normalize "part solid" -> "part-solid".
    - calcified:  true if the source explicitly calls the lesion "calcified"
                  (including "densely calcified"); false if explicitly
                  "noncalcified"; null if not stated.
    - location:   standardized lobe or region. Use exactly one of:
                  "right upper lobe", "right middle lobe", "right lower lobe",
                  "left upper lobe", "lingula", "left lower lobe",
                  "right lung", "left lung", "bilateral lungs", "lung".
    - laterality: derived from location. Use exactly one of:
                  "right" (right upper/middle/lower lobe, right lung),
                  "left"  (left upper lobe, lingula, left lower lobe, left lung),
                  "bilateral" (bilateral lungs),
                  null    (generic "lung" only).
    - noun:       "nodule" or "mass" (based on what the source uses).
                  Clusters ("multiple nodules", "nodules") count as "nodule".
    - uncertain:  true if the sentence expresses existence uncertainty
                  ("possibly", "questionable", "cannot exclude"), else false.
    - text:       the exact source sentence."""


MATCHING_CRITERIA = """\
Matching criteria:

Two nodules match across ref/pred if ALL THREE conditions hold:
1. Compatible LATERALITY — same side, OR one side has laterality null AND
   no other unmatched nodule on the determined side better fits.
   "bilateral" matches either "right" or "left" only when the predicted
   count clearly cannot be attributed to one side alone.
2. Compatible LOCATION — same lobe, or one is a parent of the other
   (e.g. "right upper lobe" matches "right lung"). Different specific
   lobes on the same side (e.g. "right upper lobe" vs "right lower lobe")
   do NOT match.
3. Compatible SIZE — both within 50% of each other, OR within 4 mm
   absolute (whichever is greater), OR at least one side has null size
   (in which case laterality + location alone determine the match).

Edge cases:
- A cluster sentence ("There are multiple ...") may match a single-nodule
  cluster on the other side (preserving the count category) or a group of
  individual nodule sentences in the same location. Use your clinical
  judgment; prefer one-to-one matches when possible.
- If several predictions fit one reference, match the closest (smallest
  size delta, then identical density, then identical calcified flag, then
  identical noun) and flag the rest as false findings.
- Mass-vs-nodule is NOT a match criterion (both nouns can match each other
  if laterality+location+size compatible), but `noun_error` is flagged
  when they differ. `noun_error` is reported but does NOT contribute to
  the composite penalty (the consolidated desiderata do not require it)."""


ERROR_TAXONOMY = """\
Per matched pair, flag each of these BOOLEANS independently:

- size_error: true if sizes are present on BOTH sides AND the difference
    exceeds the consolidated tolerance:
      * ref_size_mm < 6 mm:           tolerance +/- 1.0 mm
      * 6 mm <= ref_size_mm <= 8 mm:  tolerance +/- 1.25 mm
      * ref_size_mm > 8 mm:           tolerance +/- 15% of ref_size_mm
    If either side lacks a numeric size, size_error = false (we can't score).
- size_exact_match: true ONLY if both sides have a numeric size AND
    pred_size_mm == ref_size_mm exactly. false otherwise (including null).
- density_error: true if BOTH sides have a non-null density AND they
    disagree at the TWO-CLASS level (solid vs sub-solid, where sub-solid =
    {part-solid, ground-glass}). Same-class differences within sub-solid
    (e.g. part-solid vs ground-glass) do NOT count as density_error.
    If either side's density is null, density_error = false.
- calcified_error: true if BOTH sides have a non-null calcified flag AND
    they disagree (true vs false). If either side is null, false.
- location_error: true if the locations disagree at the lobe level, after
    allowing the parent-hierarchy equivalence above. Note: matched pairs
    already passed the laterality + lobe-compatibility filter above, so
    location_error is essentially the lobe-mismatch sub-case (e.g.
    "right upper lobe" matched a "right lung" parent — location_error = false;
    while "right upper lobe" vs "right middle lobe" should not have matched
    at all, so it does not surface here).
- noun_error: true if one side says "nodule" and the other says "mass".
    Reported for diagnostic purposes only; not penalized in the composite.
- uncertainty_error: true if one side marks `uncertain: true` and the
    other marks `false`.

For false_findings (predicted items not matched to any reference) and
missing_findings (reference items not matched to any prediction), emit only
the id(s)."""


OUTPUT_FORMAT = """\
Output format:

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "size_bucket": "6to8",
         "density": "solid", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 9, "size_bucket": "8to15",
         "density": "solid", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 9 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {
            "ref_id": "R1", "pred_id": "P1",
            "ref_size_mm": 8, "pred_size_mm": 9,
            "ref_size_bucket": "6to8", "pred_size_bucket": "8to15",
            "size_error": false, "size_exact_match": false,
            "density_error": false, "calcified_error": false,
            "location_error": false, "noun_error": false,
            "uncertainty_error": false,
            "notes": "size within 1.25 mm tolerance for 8 mm nodule"
        }
    ],
    "false_findings": ["P2"],
    "missing_findings": ["R3"]
}

`ref_size_mm` and `pred_size_mm` must be numeric (int or float) when both
sides have a size; null otherwise. `size_error` and `size_exact_match` must
both be `false` when either size is null. `ref_size_bucket` /
`pred_size_bucket` are the same bucket strings emitted in
reference_nodules / predicted_nodules; null when the corresponding
size is null.

Do NOT include any fields beyond those listed. Do NOT add commentary outside
the JSON."""


FEW_SHOT_EXAMPLES = """\
Example 1 (perfect match, size within tolerance, same density and location):

REFERENCE NODULES (PULMONARY NODULES section of the reference report):
There is an 8 mm solid nodule in the right upper lobe.

PREDICTED NODULES (PULMONARY NODULES section of the predicted report):
There is a 9 mm solid nodule in the right upper lobe.

Expected JSON:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 8, "size_bucket": "6to8",
         "density": "solid", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm solid nodule in the right upper lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 9, "size_bucket": "8to15",
         "density": "solid", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 9 mm solid nodule in the right upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 8, "pred_size_mm": 9,
         "ref_size_bucket": "6to8", "pred_size_bucket": "8to15",
         "size_error": false, "size_exact_match": false,
         "density_error": false, "calcified_error": false,
         "location_error": false, "noun_error": false,
         "uncertainty_error": false,
         "notes": "within 1.25 mm tolerance for 8 mm nodule"}
    ],
    "false_findings": [],
    "missing_findings": []
}


Example 2 (miss + false finding + matched pair with density error and
size error; calcified flag differs on one pair):

REFERENCE NODULES (PULMONARY NODULES section of the reference report):
There is a 5 mm solid nodule in the right upper lobe.
There is a 15 mm calcified nodule in the left lower lobe.

PREDICTED NODULES (PULMONARY NODULES section of the predicted report):
There is a 7 mm ground-glass nodule in the right upper lobe.
There is an 8 mm noncalcified nodule in the left upper lobe.

Expected JSON:
{
    "reference_nodules": [
        {"id": "R1", "size_mm": 5, "size_bucket": "4to6",
         "density": "solid", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 5 mm solid nodule in the right upper lobe."},
        {"id": "R2", "size_mm": 15, "size_bucket": "8to15",
         "density": null, "calcified": true,
         "location": "left lower lobe", "laterality": "left",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 15 mm calcified nodule in the left lower lobe."}
    ],
    "predicted_nodules": [
        {"id": "P1", "size_mm": 7, "size_bucket": "6to8",
         "density": "ground-glass", "calcified": false,
         "location": "right upper lobe", "laterality": "right",
         "noun": "nodule", "uncertain": false,
         "text": "There is a 7 mm ground-glass nodule in the right upper lobe."},
        {"id": "P2", "size_mm": 8, "size_bucket": "6to8",
         "density": null, "calcified": false,
         "location": "left upper lobe", "laterality": "left",
         "noun": "nodule", "uncertain": false,
         "text": "There is an 8 mm noncalcified nodule in the left upper lobe."}
    ],
    "matched_pairs": [
        {"ref_id": "R1", "pred_id": "P1",
         "ref_size_mm": 5, "pred_size_mm": 7,
         "ref_size_bucket": "4to6", "pred_size_bucket": "6to8",
         "size_error": true, "size_exact_match": false,
         "density_error": true, "calcified_error": false,
         "location_error": false, "noun_error": false,
         "uncertainty_error": false,
         "notes": "2 mm delta exceeds 1.0 mm tol for <6 mm; solid vs ground-glass"}
    ],
    "false_findings": ["P2"],
    "missing_findings": ["R2"]
}"""


IMPORTANT_NOTES = """\
Important notes:
- Only the PULMONARY NODULES content is passed to you. Do not invent other
  findings.
- Parse sizes carefully. "3.2 cm" -> 32. "2 mm micronodule" -> 2.
- For multi-dimensional measurements ("1.6 x 13 mm"), use the largest
  numeric dimension expressed in mm (here, 13).
- Treat unspecified-density nodules as `density: null`. Do NOT default to
  "solid" just because the source did not call out density.
- `calcified` is null when the source neither says calcified nor
  noncalcified; this lets the downstream metric ignore that pair for the
  calcified-accuracy KPI.
- Empty input for reference or prediction -> return the remaining side
  with all of its items in missing_findings or false_findings respectively,
  and an empty matched_pairs list.
- If both inputs are empty, return four empty lists plus empty finding lists.
- Deterministic behavior: do not rely on natural language judgment calls
  beyond the explicit criteria above. All scoring is computed from your
  JSON in downstream Python code."""


def build_prompt(ref_segment: str, hyp_segment: str) -> str:
    """Build the full user prompt.

    Parameters
    ----------
    ref_segment : str
        The PULMONARY NODULES: section content from the reference report.
        Can be empty string if the reference has no nodules.
    hyp_segment : str
        Same but for the predicted/hypothesis report.

    Returns
    -------
    str
        Complete prompt ready for LLM consumption.
    """
    ref_body = ref_segment.strip() or "(none)"
    hyp_body = hyp_segment.strip() or "(none)"

    sections = [
        OBJECTIVE,
        TEMPLATE_REF,
        NODULE_FIELDS,
        MATCHING_CRITERIA,
        ERROR_TAXONOMY,
        OUTPUT_FORMAT,
        FEW_SHOT_EXAMPLES,
        IMPORTANT_NOTES,
        f"REFERENCE NODULES (PULMONARY NODULES section of the reference report):\n{ref_body}",
        f"PREDICTED NODULES (PULMONARY NODULES section of the predicted report):\n{hyp_body}",
    ]
    return "\n\n".join(sections)
