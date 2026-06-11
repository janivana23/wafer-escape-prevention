# Wafer Quality Escape-Prevention System

An end-to-end machine-learning system for semiconductor manufacturing quality:
predicting which wafers are at risk of failing end-of-line testing ("escapes"),
monitoring for sensor drift, and surfacing failure-mode insights from
failure-analysis notes via a hybrid-retrieval RAG assistant.

Built on the public UCI SECOM dataset (1,567 production wafers, 590 sensors).

---

## Problem

In a semiconductor fab, a defective wafer that passes inspection and reaches the
customer is an **escape** — a costly quality event. Escapes are rare (~6.6% of
wafers in this dataset fail), which makes them statistically hard to catch:
a model that simply predicts "pass" for everything scores 93% accuracy while
catching zero failures. This project tackles that imbalance head-on and builds
the surrounding monitoring and analysis tooling a real quality system needs.

---

## Data

- **Source:** UCI SECOM dataset — 1,567 wafers, 590 anonymized sensor readings,
  binary pass/fail labels from end-of-line testing.
- **Class balance:** 1,463 pass / 104 fail (6.6% failure rate).
- **Cleaning:** dropped 28 sensors >50% missing and 116 zero-variance (constant)
  sensors, reducing 590 -> 446 features; remaining gaps median-imputed.
- **Failure-analysis notes:** *[YOUR WORDS — explain that the FA notes are
  synthetic, grounded in each failing wafer's real top sensor deviations, and
  why: no public wafer dataset ships with linked engineering narratives. State
  honestly that this is a simulation of the text a real fab would have.]*

---

## Approach

**Phase 1 — Data foundation.** Loaded and cleaned SECOM; established the
imbalance and missingness profile. Identified data-leakage risk and adopted a
split-then-transform discipline (all imputation/scaling fit on training data
only, enforced via scikit-learn Pipelines).

**Phase 2 — Escape-prevention classifier.** Progression from a logistic-
regression baseline -> class weighting -> Random Forest -> probability-threshold
tuning. The default 0.5 threshold produced an all-pass classifier (0% failure
recall despite 93% accuracy); tuning the threshold to ~0.12 raised failure
recall to **86%** (caught 18 of 21 test failures).

**Phase 3 — Monitoring.** Isolation Forest for unsupervised anomaly detection
and Kolmogorov–Smirnov tests for distribution drift across the production year.
**165 of 468 sensors showed statistically significant drift** (Bonferroni-
corrected), demonstrating the need for scheduled model retraining.

**Phase 4 — Text analytics.** Embedded FA notes (sentence-transformers) and
clustered them (KMeans, k selected by silhouette). Clusters separated primarily
by **excursion severity** (avg sensor deviation 5σ–9σ).

**Phase 5 — RAG root-cause assistant.** Hybrid retrieval combining semantic
search (embeddings) with keyword search (BM25), fused via normalized weighting.
Pure semantic search missed exact sensor-number queries; the hybrid layer
recovered them. Generation layer is Anthropic-API-ready with graceful
degradation.

---

## Results & honest limitations

*[YOUR WORDS — this is the most important section. Write honestly about:*
- *The precision ceiling: failure recall hit 86% but precision stayed ~11%.
  What does that tell you about the raw sensor signal? Is this a model failure
  or a data limitation?*
- *Anomaly detection caught only ~14% of failures. What does that reveal about
  the relationship between "sensor-space anomaly" and "test failure"?*
- *The clustering peaked at k=5, matching the number of generation templates.
  What's the honest caveat there?*
- *What would you do next with more time / real data?]*

---

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download SECOM (uci-secom.csv) into data/raw/
# Notebooks run in order:
#   notebooks/01_explore.ipynb   - load + clean
#   notebooks/02_model.ipynb     - classifier + threshold tuning
#   notebooks/03_anomaly.ipynb   - anomaly + drift detection
#   notebooks/04_text.ipynb      - FA notes + embedding + clustering
#   notebooks/05_rag.ipynb       - RAG assistant

# RAG module:
python -m src.rag
```

Generating FA notes via the Anthropic API requires `ANTHROPIC_API_KEY` in a
`.env` file (an offline rule-based generator is included as a fallback).

---

## What this demonstrates

ML on imbalanced data, leakage-safe pipelines, threshold tuning for asymmetric
costs, unsupervised anomaly + drift detection, text embedding and clustering,
and hybrid-retrieval RAG — implemented as reproducible notebooks plus a
refactored production module.