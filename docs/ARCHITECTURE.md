# Architecture and design decisions

This document explains *why* the system is built the way it is, what each choice
costs, and where it fails. It is written to be talked through, not just read.

```
PDFs ──► ingest ──► taxonomy tags ──► index (BM25 + FAISS)
                                            │
question ──► hybrid retrieval ──► rerank ──►│──► gate ──► grounded LLM (×1 + N samples)
                                                              │
                         verification: citations ▸ NLI ▸ consistency ▸ confidence ▸ abstain
                                                              │
                                      Streamlit UI / FastAPI ◄┘──► audit.jsonl, feedback.jsonl
```

The guiding principle: **in a safety-critical setting, an answer you cannot trace is a
liability.** So every stage either produces evidence for the user or a check on the
model, and the system is allowed to say "I don't know".

---

## 1. Ingestion: chunk on the document's own structure

**Decision.** Chunks follow Approved Document B's structure: Requirement (B1–B5) →
Section → numbered paragraph (`8.12`, or `B6` in appendices) → tables/diagrams. Each
chunk carries `volume`, `requirement`, `section_title`, `subsection`, `paragraph_ref`,
printed `page` *and* `pdf_page`.

**Why.** Engineers cite paragraphs, not token windows. A chunk that is exactly
"para 8.6" gives a citation an engineer can check in seconds. Fixed-size windows
would split paragraphs mid-sentence and make citations vague.

**How it is made reliable.**
- Headings are recognised from font size and weight (26pt section, 20pt subsection,
  15pt sub-heading) and the PDF's own table of contents.
- A paragraph number only counts if it belongs to the current section (`8.x` inside
  Section 8) and increases monotonically. This rejects "2.5m wide…" at the start of a
  line. Some paragraph numbers have no space after them in the PDF (`10.20Fire and…`);
  the pattern handles that. Result: **no gaps in paragraph numbering in either volume.**
- Running headers/footers are removed by position (top/bottom bands of the page).
- Printed page numbers differ from PDF page indices (PDF p.77 is printed p.65).
  Citations use the **printed** page because that is what a reader turns to.
- Paragraphs over ~380 words fall back to overlapping word windows (`-w1`, `-w2`).

**Two data problems found and fixed** (both would have been silent failures):
1. **Dropped ligatures.** Some embedded fonts map `fi`/`ff`/`fl` to nothing, so
   PyMuPDF returns "fre" for *fire* and "efect" for *effect* on about 1 in 10 uses
   of "fire". That silently breaks keyword search. The repair is corpus-driven: a
   word is replaced only if a ligature variant (f→fi/ff/fl/ffi) is **more frequent in
   the same corpus**. 32 word forms were repaired; every one was checked by hand,
   including non-obvious ones (*soft → soffit*, *fy → fly gallery*, *fat → flat*).
2. **Two versions of the same text.** The current editions are "collated with 2029
   amendments": Appendix B appears twice, once marked *"Effective until 1 September
   2029"* and once marked *"2029 amendment – effective from 2 September 2029"*. Both
   are indexed (both are official text) but each chunk stores its `effective_note`,
   and citing a not-yet-in-force passage triggers an automatic caveat.

**Tables.** The tables have no ruling lines, so generic table extractors produced
garbage. Each table is instead rebuilt row by row from word coordinates
(`cell | cell | cell`). This is good enough to read "3 | Office | 18 | 45" but
multi-line headers remain imperfect, so any answer citing a table gets a caveat.

## 2. Taxonomy: explicit, small, and every tag explainable

A draft two-level taxonomy (B1–B5, 2–4 sub-topics each) lives in YAML so engineers
can edit it without touching code. Tags are assigned by three rules **in order**, and
each tag records which rule produced it:

1. **structure**: the chunk's Requirement + heading matches the node (most tags);
2. **keyword**: at least 2 of the node's keywords appear;
3. **embedding**: nearest node description by cosine, only if ≥ 0.75 *and* clearly
   ahead of the runner-up.

The embedding threshold was raised after a spot check at 0.62 mis-tagged fire-doorset
paragraphs as "external walls". **An untagged chunk is better than a wrongly tagged
one**, so ~185 general/appendix chunks are deliberately left without a sub-topic.
This is a *draft for validation with senior engineers*, not a finished ontology.

## 3. Why hybrid retrieval (BM25 + dense + RRF + rerank)

- **BM25 (keywords)** is precise for the vocabulary engineers actually use:
  "REI 30", "Table 2.1", "B-s3, d2", "18m". Dense models blur such tokens.
- **Dense embeddings** (`bge-small`, local) catch paraphrase: "how far can people
  walk to an exit" ↔ "travel distance".
- **Reciprocal Rank Fusion** combines the two rankings without having to calibrate
  their incomparable scores: `score = Σ 1/(60 + rank)`.
- **Cross-encoder reranking** (`ms-marco-MiniLM`) re-reads the top 20 query–passage
  pairs jointly. It is slower but much better at relevance, and its sigmoid output is
  a usable **retrieval-strength** signal for abstention.

Everything runs locally, with no API key needed, which is what makes the keyless
retrieval-only mode possible.

**Explainability of retrieval.** Each result exposes its BM25 score/rank, dense
similarity/rank, RRF score, fused rank, rerank relevance, and the query terms it
matched (highlighted in the UI). "Why did it show me this?" always has an answer.

## 4. Grounded generation

- The LLM sees only the retrieved passages, each labelled with its chunk ID,
  citation and validity note.
- It must return JSON (validated with Pydantic): an interpretation of the question,
  answer sentences each with `citations` and a short **verbatim quote**, an
  **evidence-linked reasoning summary**, caveats, and an `abstained` flag.
- The reasoning pathway is a summary *written for the reader* that links each step to
  evidence IDs. **It is not the model's hidden reasoning**, and the UI says so.
- Providers are pluggable (OpenAI by default, Ollama locally) behind a one-method
  interface; the default model is `gpt-4.1-mini` (small, inexpensive, fast; see the README for why not a newer model).

## 5. The reliability layer: why each check exists

| Check | Question it answers | Catches |
|---|---|---|
| Citation validation | Does the cited passage exist in what was retrieved? Is the quote really in it? | Invented chunk IDs, fabricated or paraphrased "quotes" |
| NLI faithfulness | Does the cited text actually *entail* this sentence? | Correct-looking citations attached to claims they don't support |
| Self-consistency | Would the model say the same thing again? | Under-determined answers that flip between samples |
| Retrieval strength | Did we find anything relevant at all? | Out-of-scope questions |

### Why an NLI check (and not "ask the LLM if it was faithful")
A separate, small, local **NLI cross-encoder** (`nli-deberta-v3-small`) judges each
sentence against its own citations. It is independent of the generator (it does not
share its blind spots), deterministic, cheap, and its output is a probability that
can be shown. The sentence's score is its best entailment over a set of short
"premise windows" from the cited passages.

**What real output taught us** (each found by inspecting failures, each now tested):
1. **Short premises.** This NLI model labels a sentence against *itself* as entailment,
   but adding one unrelated sentence to the premise flips it to "neutral". Premise
   windows are therefore 1, 2 and 3 consecutive sentences, not just 3.
2. **Lists.** Approved Document B states conditions as a lead-in ("…if either of the
   following applies.") plus lettered items. NLI could not link item *b* back to its
   lead-in, so each item is also paired directly with its lead-in sentence.
3. **Provenance, but optional.** "In Volume 1, blocks of flats … should be fitted with
   sprinklers" scored 0.00 because the passage never says it is from Volume 1. Adding
   the provenance the LLM sees ("In Approved Document B Volume 1 (Dwellings) – Section 7
   …, paragraph 7.4: …") fixes that (1.00), but it *hurt* a plain restatement of para
   1.1 (0.78 → 0.05). Each window is now scored with and without the provenance and
   the better score is kept.
4. **Numbers.** NLI is weak with numbers: in an offline perturbation test it accepted
   23% of sentences whose numbers had been doubled ("1100 mm" → "2200 mm"). In fire
   safety the numbers *are* the guidance, so a **deterministic numeric guard** marks a
   sentence Unsupported if any of its numbers is missing from the cited passages. After
   the guard: 0% of number-swapped claims accepted, true sentences unaffected.
5. **Tables.** NLI cannot read rows like `3 | Office | 18 | 45`. A transparent rule
   upgrades such sentences to **Partially supported** (never Supported) when the quote
   is verbatim *and* every number appears in the cited table; the UI says so.

**Guarding against "fixing" the verifier into leniency.** Every change above makes it
easier to accept a sentence, so `eval/perturbation_check.py` re-verifies deliberately
false variants of every accepted sentence (numbers doubled; "not" inserted) against the
same citations. Final result: 0% of number swaps and 10% of negations were still
labelled Supported. The negation weakness is a known limit of the small NLI model.
Because these fixes were found on the evaluation set itself, the final evaluation
figures are optimistic; the README shows the before/after history.

### Why self-consistency
Sampling N=5 extra answers at temperature 0.7 and comparing them to the main answer
(semantic similarity + overlap of cited chunks + agreement on abstaining) is a cheap
uncertainty signal that needs no labelled data. **Limitation:** consistency is not
correctness; a model can be consistently wrong, which is why it is only 30% of the
score and can never override faithfulness.

## 6. Confidence and abstention: how the thresholds were chosen

```
overall = 0.30·retrieval_strength + 0.40·faithfulness + 0.30·consistency
High    = overall ≥ 0.75 AND faithfulness ≥ 0.75 AND no Unsupported sentence
Medium  = overall ≥ 0.50 AND faithfulness ≥ 0.50
Low     = otherwise
Abstain = retrieval < 0.10 (before calling the LLM) | model abstains |
          no valid citation | faithfulness = 0 | overall < 0.35
```

- **Faithfulness has the biggest weight** because it is the only signal that checks
  the answer against the evidence. The hard caps mean strong retrieval and high
  consistency can **never** compensate for an answer its own citations don't support.
- **The retrieval gate (0.10) is deliberately low.** It is a coarse filter for
  clearly off-topic questions (electrical cables, cake recipes score ≈ 0.00). The
  evaluation includes a sensitivity table: a gate of 0.9 would refuse all five
  out-of-scope questions while passing all in-scope ones *on this set*, but that would
  be tuning a threshold on 25 hand-written questions, and real users phrase things
  less cleanly. Subtle out-of-scope questions (e.g. U-values, which share vocabulary
  with external-wall guidance) are left to the LLM's instruction to abstain and to
  the verification layer.
- **NLI bands** (≥0.5 Supported, ≥0.15 Partial) and the confidence cut-offs are
  conventional starting points, not fitted values. Calibrating them needs
  expert-labelled answers; that is the first item of real pilot work.
- Abstentions show **no** confidence label (a "High" label on a refusal is
  meaningless), but the signals are still recorded for audit.

## 7. Governance

- **Audit log** (`logs/audit.jsonl`): timestamp, question, retrieved chunk IDs and
  scores, provider, model, prompt version, answer with per-sentence faithfulness,
  confidence breakdown, abstention flag and reason, latency. When an answer is
  withheld, the withheld draft is kept in the log so refusals can be reviewed.
- **Data minimisation**: no user IDs, IP addresses or sessions are recorded; emails,
  UK phone numbers and postcodes typed into questions or comments are redacted.
- **Prompt versioning**: every log line records which prompt produced it.
- These support transparency and record-keeping principles (as in the EU AI Act and
  GDPR). **This is not a claim of legal compliance.**

## 8. What fails, and why

| Failure | Cause | Mitigation / next step |
|---|---|---|
| Wrong neighbour retrieved (eval q03: para 2.6 instead of 2.5) | Adjacent paragraphs differ by one phrase ("one storey" vs "two or more storeys") | Follow cross-references ("comply with paragraphs 2.5 and 2.6") and add neighbouring paragraphs to the context |
| NLI under-scores long, multi-claim sentences (eval q16: a correct answer was withheld) | One premise window rarely entails a sentence that merges a lead-in and several list items | Prompt asks for one claim per sentence; splitting sentences into atomic claims before NLI is a next step |
| NLI still accepts some negated claims (10% in the perturbation test) | Small NLI models are weak at negation | Shown as a limitation; a larger NLI model or an LLM-as-judge second opinion are options, at a latency/cost price |
| Table answers capped at Partially supported | NLI cannot read tables | Numeric grounding rule; structured table extraction is a next step |
| Subtle out-of-scope questions pass retrieval | Shared vocabulary (U-value ↔ external wall) | Relies on LLM abstention + verification; an out-of-scope classifier is a next step |
| Diagram content is mostly lost | Diagrams are images; only labels are extractable | Captions are indexed so diagrams can be cited and found in the PDF |
| macOS: torch + faiss bundle separate OpenMP runtimes | Packaging of the pip wheels | On macOS only, the duplicate runtime is allowed, faiss is single-threaded, and faiss results are checked against an exact NumPy search every time the index loads; Linux (Docker, CI) is unaffected |
| Evaluation set is small and not expert-reviewed, and the verifier was tuned on it | One-day build | Treat numbers as illustrative; expand with engineers and keep a held-out set |
| Free-tier API limits (50 requests/day/model) | Account tier | The evaluation is resumable and records every LLM call so verification changes can be re-scored offline (`--replay`) |
