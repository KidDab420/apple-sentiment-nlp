"""
absa.py
Aspect-Based Sentiment Analysis.

Approach:
  1. Extract candidate aspect terms with spaCy (noun chunks / frequent nouns).
  2. For each occurrence, score the sentiment of the surrounding context window
     with VADER (a lightweight, transparent baseline for ABSA).
  3. Aggregate to produce an aspect -> sentiment-distribution map.

The same context-window idea is what a fine-tuned BERT-ABSA model learns
end-to-end; this module gives an interpretable baseline + the visualisations.
"""
import warnings
warnings.filterwarnings("ignore")

from collections import Counter, defaultdict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from data_prep import build_clean_dataframe

NLP = spacy.load("en_core_web_sm", disable=["ner"])
VADER = SentimentIntensityAnalyzer()

# Domain stop-aspects: too generic / not real product aspects
ASPECT_STOP = {"apple", "aapl", "iphone", "ipad", "ipod", "mac", "amp",
               "today", "day", "thing", "people", "one", "time", "year",
               "way", "lot", "guy", "everyone", "someone", "rt", "http"}


def discover_aspects(df, top_n=12, min_count=15):
    """Find the most frequent noun aspects across the corpus."""
    counter = Counter()
    for doc in NLP.pipe(df["parse_text"].tolist(), batch_size=64):
        for chunk in doc.noun_chunks:
            root = chunk.root
            if root.pos_ in {"NOUN", "PROPN"} and root.is_alpha and len(root) > 2:
                lemma = root.lemma_.lower()
                if lemma not in ASPECT_STOP:
                    counter[lemma] += 1
    aspects = [a for a, c in counter.most_common() if c >= min_count][:top_n]
    return aspects, counter


def aspect_sentiment(df, aspects, window=4):
    """For each aspect occurrence, score sentiment of a +/- window of tokens."""
    records = defaultdict(lambda: Counter())
    for doc in NLP.pipe(df["parse_text"].tolist(), batch_size=64):
        toks = [t for t in doc]
        lemmas = [t.lemma_.lower() for t in toks]
        for i, lem in enumerate(lemmas):
            if lem in aspects:
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                ctx = " ".join(t.text for t in toks[lo:hi])
                c = VADER.polarity_scores(ctx)["compound"]
                pol = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
                records[lem][pol] += 1
    rows = []
    for asp, cnt in records.items():
        total = sum(cnt.values())
        rows.append({"aspect": asp, "total": total,
                     "positive": cnt["positive"], "neutral": cnt["neutral"],
                     "negative": cnt["negative"],
                     "net_sentiment": round((cnt["positive"] - cnt["negative"]) / total, 3)})
    return pd.DataFrame(rows).sort_values("total", ascending=False)


def plot_absa(absa_df):
    d = absa_df.set_index("aspect")[["negative", "neutral", "positive"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    d.plot(kind="barh", stacked=True, ax=ax,
           color=[config.SENT_COLORS["negative"], config.SENT_COLORS["neutral"],
                  config.SENT_COLORS["positive"]])
    ax.set_title("Aspect-Based Sentiment — Apple Tweets", fontweight="bold")
    ax.set_xlabel("Number of mentions")
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "09_absa_aspect_sentiment.png", bbox_inches="tight")
    plt.close(fig)

    # Net-sentiment lollipop
    s = absa_df.sort_values("net_sentiment")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [config.SENT_COLORS["positive"] if v >= 0 else config.SENT_COLORS["negative"]
              for v in s["net_sentiment"]]
    ax.hlines(y=s["aspect"], xmin=0, xmax=s["net_sentiment"], color=colors, alpha=0.6)
    ax.scatter(s["net_sentiment"], s["aspect"], color=colors, s=80, zorder=3)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Net Sentiment by Aspect  (pos − neg) / total", fontweight="bold")
    ax.set_xlabel("Net sentiment score")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "10_absa_net_sentiment.png", bbox_inches="tight")
    plt.close(fig)


def run_absa(df=None):
    if df is None:
        df = build_clean_dataframe()
    aspects, counter = discover_aspects(df)
    print("Discovered aspects:", aspects)
    absa_df = aspect_sentiment(df, set(aspects))
    absa_df.to_csv(config.RES_DIR / "absa_aspect_sentiment.csv", index=False)
    plot_absa(absa_df)
    print("\nAspect-level sentiment:")
    print(absa_df.to_string(index=False))
    return absa_df


if __name__ == "__main__":
    run_absa()
