# Ex3 negative-generation: human annotation guidelines

You've been given `human_annotation_sample.csv` -- 100 rows chosen uniformly
at random (see `src/verify_negatives.py`) from the generated DPO negative
pairs for the "ex 3" cognitive-distortion experiment (outline.md). This is
one of the two verification steps for that dataset; the other is an
automated LLM-judge pass over all rows. This sample is the only place a
human eye checks the data before it's used to train a model, so be
skeptical rather than charitable when a row is borderline.

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

An automated LLM-judge (see `src/verify_negatives.py`) already looked at
every row and picked a winner between `chosen` and `rejected`. You are
answering **the exact same question, independently**, so your label and the
judge's can be compared directly (see `src/compare_judgments.py`):

**Given the client statement, which response is the better therapist
response overall -- `chosen` or `rejected` -- or are they about equally
good (`tie`)?**

Judge it as a reader would, not by checking off whether the intended
ablation technically happened -- but the component descriptions below tell
you what `rejected` was *supposed* to get wrong, which is usually exactly
why it's worse (or, if the ablation didn't take, why it might not be).

## The four components

Only judge the row against the failure mode named in `component_ablated` --
don't penalize `rejected` for unrelated flaws.

**`sq_ablation`** -- Socratic Questioning removed, replaced with unsolicited
advice-giving. `rejected` should contain no open-ended questions or Socratic
dialogue at all, and instead tell the client directly what to think, do, or
feel. *Ablation didn't take if:* `rejected` still asks the client questions,
still invites the client to reflect/explore on their own, or is only
slightly more directive than `chosen`.

**`ev_ablation`** -- Empathetic Validation removed, replaced with cold,
mechanical clinical logic. `rejected` should contain no emotional
acknowledgment or warmth, jumping straight into analyzing the client's
thinking like a technical problem. *Ablation didn't take if:* `rejected`
still opens with or contains sympathy, validation, or warm language
("that sounds hard," "it makes sense you'd feel that way," etc.), even if
softened.

**`cr_toxic_positivity`** -- Cognitive Reframing replaced with toxic
positivity / dismissal. `rejected` should push an unrealistically cheerful,
minimizing, or dismissive reframe that brushes off the client's actual
problem, rather than a realistic, evidence-based alternative perspective.
*Ablation didn't take if:* `rejected` still reads as a grounded, realistic
reframe rather than forced positivity, or the "positivity" is mild enough
that it could plausibly appear in a legitimate response.

**`cr_distortion_reinforcement`** -- Cognitive Reframing replaced with
covert reinforcement of the client's distortion. `rejected` should sound
supportive/empathetic on the surface but actually agree with or validate
the client's distorted thinking (e.g. agreeing a catastrophic prediction is
likely, agreeing with their mind-reading of others). *Ablation didn't take
if:* `rejected` still challenges the distortion, offers a counter-perspective,
or hedges enough that it's not really agreeing with the distortion.

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

Fill in **only** `human_label` and `human_notes`. Leave every other column
untouched -- don't reorder or delete rows.

**`human_label`** -- exactly one of these three strings (lowercase, matching
the judge's own output):

| label | use when |
|---|---|
| `chosen` | `chosen` is the better response overall |
| `rejected` | `rejected` is the better response overall (rare -- flag it, it means generation went backwards) |
| `tie` | genuinely about equally good, or you can't meaningfully separate them |

**`human_notes`** -- free text, one short sentence. Always worth a note when
you pick anything other than `chosen`, or when `chosen` wins only barely --
say briefly *why* (e.g. "ablation didn't take, still asks questions",
"malformed -- empty response", "chosen is generic here too", "toxic
positivity is mild but present"). Not required, but useful, on clean
`chosen` calls.

## Practical notes

- Judge each row independently -- don't let your read of one row about a
  client statement bias your read of another row for the same statement
  under a different component.
- This is synthetic CBT-Bench practice data, not real client records;
  treat the content professionally but there's no privacy concern in
  discussing specifics in `human_notes`.
- Save the file as CSV (UTF-8), same filename, when you're done with all
  100 rows.
