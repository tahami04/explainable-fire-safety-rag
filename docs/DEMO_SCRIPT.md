# 3-minute demo script

**Setup (before the demo):** `streamlit run app.py` with `OPENAI_API_KEY` set, browser at
<http://localhost:8501>, sidebar on defaults (no taxonomy filter, 5 consistency samples).
Each question takes roughly 5–15 seconds in full mode.

The behaviour described below is what the system produced in the recorded evaluation
run (`eval/results/full.json`); generations can vary slightly between runs.

---

## 0:00 – 0:20 · The framing

> "In safety-critical engineering, an AI answer you can't trace is a liability. This
> assistant answers questions about the UK fire-safety guidance, Approved Document B,
> and it's built so that every claim is traceable to a paragraph, page and quote,
> every answer is checked, and it is allowed to say *I don't know*."

Point at the disclaimer banner: this is a demonstration, not professional advice.

## 0:20 – 1:10 · 1) A clear, well-supported answer

Ask: **"When should a block of flats be fitted with a sprinkler system?"**

What to point at:
1. **Answer sentences with citation chips:** `[Vol 1, B3, para 7.4, p. 60]`. "That's
   the exact paragraph and the printed page, so an engineer can check it in seconds."
2. **Green "✔ Supported" label on each sentence.** "A separate local NLI model checked
   that the cited text actually *entails* each sentence. It doesn't trust the LLM's
   word for it."
3. **Confidence: High**, then open the breakdown: retrieval strength ≈ 1.00,
   faithfulness 1.00, self-consistency ≈ 0.97. "Not one opaque number. You can see
   why it is confident: the evidence is strongly relevant, every sentence is
   supported, and five resampled answers agreed."
4. If a sentence shows *"A quote was not found verbatim"*: "Even when the claim is
   supported, the system tells you the model paraphrased its quote."
5. Open **Evidence #1**: highlighted query terms, and the *Why this was retrieved*
   table (BM25 rank, semantic rank, fused rank, reranker relevance).

## 1:10 – 2:10 · 2) A partly supported answer

Ask: **"In a dwelling building with a storey 18m or more in height, what class must
insulation products in the external wall achieve?"**

What to point at:
1. Sentences are labelled **◐ Partially supported**, so **Confidence is Medium**, not
   High. "High confidence requires strong faithfulness. Partial support caps it."
2. The **caveats**: the answer depends on whether *regulation 7(2)* applies, which is
   stricter. "This is where a junior engineer could get it wrong. The system surfaces
   the condition instead of hiding it."
3. **Reasoning pathway**, labelled as an *evidence-linked reasoning summary*: each
   step lists the chunk IDs it relies on. "This is a summary written for you and
   linked to evidence. I don't present it as the model's hidden thinking."
4. Optionally: *"Table look-ups (e.g. travel distance in Table 2.1) are capped at
   Partially supported because NLI can't read table rows; the UI says exactly why."*

## 2:10 – 2:50 · 3) An out-of-scope refusal

Ask: **"What size of electrical cable should I use for a cooker circuit?"**

What to point at:
1. Red box: **"Insufficient evidence in the indexed guidance; refer to a competent fire
   engineer."**
2. **Why:** best passage relevance 0.00 is below the threshold, so the LLM was never
   called. "Refusing is a feature. It's the safe default when the guidance doesn't
   cover the question."
3. Optional second example: **"What U-value must a new external wall achieve?"**
   Retrieval finds external-wall passages (relevance ≈ 0.83), so the gate lets it
   through, and then the model itself abstains. "Two layers of refusal."

## 2:50 – 3:00 · Close

> "Every query is written to an audit log (question, evidence IDs and scores, model,
> prompt version, confidence breakdown, abstention), with personal data redacted, and
> the thumbs up/down feeds a pilot feedback log. The evaluation is in the README:
> real numbers, including where it fails."

Click 👍 to show the feedback capture.
