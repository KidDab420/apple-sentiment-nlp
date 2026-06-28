# Apple Twitter Sentiment Analysis — NLP Pipeline
**CDS6344 Social Media Computing** · End-to-end sentiment, opinion mining & ABSA

An end-to-end Natural Language Processing pipeline that extracts, analyses and
visualises sentiment, opinions and aspect-based sentiment from ~3,800 real
tweets about Apple ($AAPL / @apple). It combines classical machine learning,
lexicon methods, dependency-based opinion mining, aspect-based sentiment
analysis, and (in the companion notebook) deep-learning and transformer models.

---

## 1. Project structure

```
apple-sentiment-nlp/
├── data/
│   └── Apple-Twitter-Sentiment-DFE.csv      # raw dataset (3,886 tweets)
├── src/
│   ├── config.py            # paths, label map, constants
│   ├── data_prep.py         # loading, cleaning, lemmatisation
│   ├── eda.py               # class dist, word clouds, top terms
│   ├── classical_ml.py      # NB, LogReg, SVM, RandomForest + evaluation
│   ├── opinion_mining.py    # VADER baseline + dependency opinion triples
│   ├── absa.py              # aspect extraction + aspect-level sentiment
│   └── run_pipeline.py      # runs every stage above
├── notebooks/
│   └── deep_learning_transformers.ipynb   # BiLSTM/CNN/GRU + BERT (GPU)
├── outputs/
│   ├── figures/             # all generated charts (.png)
│   └── results/             # metrics tables, reports, summary JSON
├── requirements.txt
└── README.md
```

## 2. Dataset

| Property | Value |
|---|---|
| Source | Figure-Eight / CrowdFlower "Apple Twitter Sentiment" (Data for Everyone) |
| Tweets | 3,886 raw → 3,798 after cleaning |
| Labels | `1`=negative, `3`=neutral, `5`=positive (`not_relevant` dropped) |
| Distribution | neutral 2,157 · negative 1,219 · positive 422 (imbalanced) |
| Extra fields | `sentiment:confidence`, `date`, `query`, `id` |

## 3. Setup

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 4. How to run

**Classical / lexicon / opinion / ABSA (CPU, ~1 min):**
```bash
cd src
python run_pipeline.py          # runs all five stages, writes figures + results
```
Individual stages can also be run on their own, e.g. `python classical_ml.py`.

**Deep learning + transformers (GPU):**
Open `notebooks/deep_learning_transformers.ipynb` in **Google Colab**, set
`Runtime ▸ Change runtime type ▸ GPU`, upload the CSV when prompted, *Run all*.

## 5. Methodology (mapped to the pipeline)

1. **Pre-processing** — HTML unescape, lowercasing, URL/@mention/#hashtag handling,
   non-alpha removal, stop-word removal (negations kept), WordNet lemmatisation.
2. **Feature engineering** — TF-IDF (uni+bigrams, sublinear, 8k features).
3. **Classical ML** — Naive Bayes, Logistic Regression, Linear SVM, Random Forest
   with balanced class weights and 5-fold cross-validation.
4. **Opinion mining** — VADER lexicon baseline + spaCy dependency parsing to
   extract `(target, opinion, polarity)` triples.
5. **ABSA** — spaCy noun-chunk aspect discovery + context-window VADER scoring →
   per-aspect sentiment distribution and net-sentiment.
6. **Deep learning / Transformers** — BiLSTM, CNN, GRU, and fine-tuned
   DistilBERT/BERT (notebook).

## 6. Headline results (test set, macro-F1)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.70 | **0.63** |
| Linear SVM | 0.71 | 0.62 |
| Random Forest | **0.73** | 0.61 |
| Naive Bayes | 0.70 | 0.54 |
| VADER (lexicon baseline) | 0.55 | 0.52 |

*Transformer fine-tuning (DistilBERT/BERT) typically lifts macro-F1 further —
run the notebook to obtain those numbers on the GPU.*

**Aspect insights:** `app` is net-positive (+0.37); `store` (−0.45) and the
`protest`/`anger`/`nyc` cluster (−0.9 to −1.0) are strongly negative. Several
high-frequency neutral aspects (`computer`, `studio`, `outlet`) come from
repeated promotional/near-duplicate tweets — a data-quality observation.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `OSError: Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| `UnicodeDecodeError` reading the CSV | the loader uses `encoding='latin-1'`; keep it |
| `LookupError` from NLTK | resources auto-download on first run; needs internet |
| Transformer cell is very slow | switch Colab runtime to GPU; reduce epochs/batch |
| Different numbers than the table | expected — depends on library versions & seeds |

## 8. FAQ

**Why is neutral the majority class?** Many tweets are factual/news ($AAPL stock
chatter), which annotators marked neutral. We use balanced class weights and
report macro-F1 so minority classes count equally.

**Why macro-F1 instead of accuracy?** Accuracy is inflated by the dominant
neutral class; macro-F1 better reflects performance across all three sentiments.

**Why VADER for opinion/ABSA scoring?** It is purpose-built for social-media text
and gives a transparent, reproducible baseline. A fine-tuned BERT-ABSA model can
replace the VADER scorer for higher accuracy.
