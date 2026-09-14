# Ex3 negative-generation: human annotation guidelines

You've been given `human_annotation_sample.csv` -- 100 rows chosen uniformly
at random (see `src/llm-as-judge/verify_negatives.py`) from the generated
DPO negative pairs for the "ex 3" cognitive-distortion experiment
(outline.md). This is one of the two verification steps for that dataset;
the other is an automated LLM-judge pass over all rows. This sample is the
only place a human eye checks the data before it's used to train a model,
so be skeptical rather than charitable when a row is borderline.

## What you're looking at

Each row is a `(client_statement, component_ablated)` pair with two candidate
therapist responses to the same client statement:

| column | meaning |
|---|---|
| `qid` | row id, for reference back to the source data |
| `exercise` / `level` | CBT-Bench exercise metadata, not relevant to the judgment |
| `component_ablated` | which of the 4 CBT components this row's `rejected` was generated to violate (see below) |
| `client_statement` | the client's message |
| `chosen` (y_w) | the original CBT-Bench ground-truth therapist response -- assumed good |
| `rejected` (y_l) | an LLM-rewritten version of `chosen`, deliberately generated to be flawed in the specific way named by `component_ablated` |

`chosen` is identical across all 4 components for a given client statement;
only `rejected` changes. You are judging **`rejected`**, using `chosen` as
the reference point for what a good response looks like.

## The question you're answering

An automated LLM-judge (see `src/llm-as-judge/verify_negatives.py` and the
per-component prompts in `src/llm-as-judge/prompts.md`) already looked at
every row and decided whether the intended ablation actually took. You are
answering **the exact same question, independently, for the specific
`component_ablated` named in the row**, so your label and the judge's can
be compared directly (see `src/compare_judgments.py`):

**Does `rejected` (y_l) successfully exhibit the failure mode named by
`component_ablated`, relative to `chosen` (y_w) as the reference for what a
good response looks like -- yes (a VALID negative) or no (INVALID, the
ablation didn't take)?**

This is a check of whether the intended ablation technically happened, not
a general "which response is better" preference call -- the component
descriptions below give the specific pass/fail condition for each of the
four failure modes.

## The four components

Only judge the row against the failure mode named in `component_ablated` --
don't penalize `rejected` for unrelated flaws. These are the same pass/fail
conditions given to the LLM judge in `src/llm-as-judge/prompts.md`.

**`sq_ablation`** (Socratic Questioning Ablation) -- VALID if `rejected`
contains ZERO open-ended Socratic questions aimed at guided discovery, and
instead gives direct, unsolicited advice, commands, or directive telling.
*INVALID (ablation didn't take) if:* `rejected` still asks the client
questions, still invites the client to reflect/explore on their own, or is
only slightly more directive than `chosen`.

**`ev_ablation`** (Empathetic Validation Ablation) -- VALID if `rejected`
COMPLETELY STRIPS OUT all emotional validation, warmth, and relational
empathy, jumping straight into mechanical analysis or clinical logic that
ignores the client's emotional state. *INVALID if:* `rejected` still opens
with or contains sympathy, validation, or warm language ("that sounds
hard," "it makes sense you'd feel that way," etc.), even if softened.

**`cr_toxic_positivity`** (Toxic Positivity / Dismissive Reframing) --
VALID if `rejected` pushes superficial optimism, forced cheerfulness, or
dismissive reassurance (e.g. "Just look on the bright side!") that
minimizes the client's real distress, without guiding them toward a
realistic, evidence-based alternative. *INVALID if:* `rejected` still reads
as a grounded, realistic reframe rather than forced positivity, or the
"positivity" is mild enough that it could plausibly appear in a legitimate
response.

**`cr_distortion_reinforcement`** (Distortion Reinforcement) -- VALID if
`rejected` actively reinforces, validates, or agrees with the client's
maladaptive cognitive distortion (e.g. confirming a catastrophic
prediction, agreeing with mind-reading, validating all-or-nothing
thinking), often while sounding superficially supportive. *INVALID if:*
`rejected` still challenges the distortion, offers a counter-perspective,
or hedges enough that it's not really agreeing with the distortion.

Regardless of component, `rejected` should also stay grammatically fluent,
coherent, and relevant to the client statement -- an ablation that reads as
incoherent noise isn't a valid negative even if it technically avoids the
named failure mode.

## Other things to flag regardless of component

- **Empty or truncated**: `rejected` is blank, cut off mid-sentence, or
  clearly incomplete.
- **Refusal**: `rejected` is the model declining to answer ("I can't help
  with that," "as an AI...") rather than an in-character therapist response.
- **Identical or near-identical to `chosen`**: the rewrite barely changed
  anything, so there's no real contrast for the preference pair.
- **Off-topic or incoherent**: `rejected` doesn't actually respond to
  `client_statement`, or doesn't read as a therapist response at all.
- **`chosen` itself looks weak**: rare, but if the "ground truth" response
  looks generic, off-topic, or arguably worse than `rejected`, flag it --
  that's a data issue upstream of generation, not a judgment about
  `rejected`.

## How to fill in the two blank columns

Fill in **only** `human_valid` and `human_notes`. Leave every other column
untouched -- don't reorder or delete rows.

**`human_valid`** -- exactly one of these two strings (lowercase, matching
the judge's own `is_valid_negative` field):

| label | use when |
|---|---|
| `true` | `rejected` successfully exhibits the failure mode named by `component_ablated` (a valid negative) |
| `false` | the ablation didn't take -- `rejected` doesn't actually exhibit that failure mode (rare -- flag it, it means generation went backwards), or `rejected` is malformed/incoherent/off-topic |

**`human_notes`** -- free text, one short sentence. Always worth a note when
you mark `false`, or when the ablation only barely took -- say briefly *why*
(e.g. "ablation didn't take, still asks questions", "malformed -- empty
response", "toxic positivity is mild but present"). Not required, but
useful, on clean `true` calls.

## Practical notes

- Judge each row independently -- don't let your read of one row about a
  client statement bias your read of another row for the same statement
  under a different component.
- This is synthetic CBT-Bench practice data, not real client records;
  treat the content professionally but there's no privacy concern in
  discussing specifics in `human_notes`.
- Save the file as CSV (UTF-8), same filename, when you're done with all
  100 rows.
