# AI Support Agent for AmazonHelp

An AI system that classifies incoming AmazonHelp customer support messages into intents, drafts a
reply grounded in how Amazon has historically resolved similar issues, and decides whether
each message should be auto-handled or escalated to a human with a stated reason.

See `REPORT.md` for full problem framing, results, failure analysis, and the "what's misleading
about my headline number" discussion. See `decision.md` for the full decision log.

## Project structure

```
data/                          # datasets (raw + processed; twcs.csv itself is not committed, see below)
src/
  prepare_dataset.py           # builds customer<->reply conversation pairs from the raw dataset
  run_classifier.py            # classifies the pairs corpus into intents (one-time corpus build)
  build_retrieval.py           # builds local embeddings for RAG retrieval
  classify_intent.py           # intent classification logic (batched LLM calls via Groq)
  generate_reply.py            # RAG retrieval + grounded reply generation
  escalate.py                  # rule-based auto-handle vs. escalate decision logic
  run_agent.py                 # unified pipeline: classify -> escalate/reply
  baseline_trivial.py          # trivial baseline (always predicts most common intent)
  baseline_simple.py           # simple baseline (TF-IDF + Logistic Regression)
  build_golden_set.py          # samples the 250-candidate golden evaluation set
  evaluate.py                  # computes headline accuracy metrics vs. both baselines
  llm_judge.py                 # LLM-as-judge for reply quality (relevance/tone/resolution)
  run_judge_eval.py            # runs the judge across a sample + prepares human-comparison file
REPORT.md                      # full report (problem framing, results, failure analysis, etc.)
decision.md                    # decision log
```

## Setup 

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1        # Windows PowerShell
   # source venv/bin/activate       # Mac/Linux
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your own API key(s):
   ```
   GROQ_API_KEY=your_groq_key_here
   ```
   A free key can be created at https://console.groq.com. This project uses Groq's
   `openai/gpt-oss-120b` model for classification, reply generation, and judging.

3. The raw dataset (`data/twcs.csv`, ~516MB, the Kaggle "Customer Support on Twitter" dataset) is
   **not included in this repo** due to size. Download it from
   https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter, extract `twcs.csv`,
   and place it at `data/twcs.csv`. **This is only needed if you want to rebuild the corpus from
   scratch (Step A below) — it is not needed to reproduce the headline results (Step B), since the
   already-processed, subsampled files are committed to this repo.**

## Reproducing headline results 

The one-time corpus build (extracting and classifying conversation pairs from the raw 3M-row
dataset) is **already done** and its outputs are committed to this repo
(`data/amazon_pairs_classified.csv`, `data/pair_embeddings.npy`, `data/golden_eval_raw.csv` with
labels, `data/baseline_model.pkl`). You do not need to regenerate these to verify results, doing
so would involve re-running language detection and LLM classification calls that are not
necessary just to check the headline numbers, and would risk hitting free-tier API rate limits
unnecessarily.

**To reproduce the headline accuracy numbers (intent classification vs. both baselines, plus
escalation accuracy):**
```
python src/evaluate.py
```
This runs your classifier, the trivial baseline, and the simple baseline against the 164
hand-labeled golden examples, and prints:
- Intent classification accuracy for all three approaches
- Escalation decision accuracy
- Saves detailed per-example results to `data/evaluation_results.csv`

**To reproduce the LLM-judge reply-quality evaluation:**
```
python src/run_judge_eval.py
```
This generates grounded replies and judge scores for a 25-example sample, saving to
`data/judge_vs_human.csv` (human scores for the same 25 examples are already filled in this
committed file, so agreement can be checked without re-labeling).

**To try the full pipeline on a single message:**
```
python src/run_agent.py "Where is my package? It's 3 days late."
```

## Known limitations

See `REPORT.md`, Section 4 "What Is Misleading About My Headline Number?" for a full,
honest discussion of what the headline accuracy numbers do.