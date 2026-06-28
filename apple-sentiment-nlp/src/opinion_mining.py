"""
opinion_mining.py
(1) Lexicon-based sentiment baseline using VADER (built for social media text).
(2) Explicit opinion extraction via spaCy dependency parsing:
    adjective (opinion word) -> noun (opinion target) triples.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config
from data_prep import build_clean_dataframe

NLP = spacy.load("en_core_web_sm", disable=["ner"])
VADER = SentimentIntensityAnalyzer()
sns.set_theme(style="whitegrid")


def vader_label(text):
    c = VADER.polarity_scores(text)["compound"]
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def run_vader_baseline(df):
    """Lexicon baseline on the ORIGINAL text (VADER handles emoji/slang/caps)."""
    df = df.copy()
    df["vader_pred"] = df["text"].apply(vader_label)
    acc = accuracy_score(df["label"], df["vader_pred"])
    f1 = f1_score(df["label"], df["vader_pred"],
                  average="macro", labels=config.LABEL_ORDER)
    cm = confusion_matrix(df["label"], df["vader_pred"], labels=config.LABEL_ORDER)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=config.LABEL_ORDER, yticklabels=config.LABEL_ORDER, ax=ax)
    ax.set_title(f"VADER Lexicon Baseline\nacc={acc:.2f}, macroF1={f1:.2f}",
                 fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "08_vader_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print(f"VADER baseline: accuracy={acc:.3f}  macroF1={f1:.3f}")
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def extract_opinions(text):
    """Return (target_noun, opinion_adjective, polarity) triples from one tweet.
    Only adjectives that carry sentiment polarity are kept as opinions."""
    doc = NLP(text)
    triples = []
    for token in doc:
        if token.pos_ != "ADJ":
            continue
        # adjectival modifier: 'great phone' -> (phone, great)
        if token.head.pos_ in {"NOUN", "PROPN"} and token.dep_ == "amod":
            triples.append((token.head.lemma_.lower(), token.lemma_.lower()))
        # copula: 'battery is amazing' -> subject is the target
        if token.dep_ == "acomp":
            for s in token.head.children:
                if s.dep_ in {"nsubj", "nsubjpass"} and s.pos_ in {"NOUN", "PROPN"}:
                    triples.append((s.lemma_.lower(), token.lemma_.lower()))
    out = []
    for target, opinion in triples:
        pol = VADER.polarity_scores(opinion)["compound"]
        if pol >= 0.05:
            polarity = "positive"
        elif pol <= -0.05:
            polarity = "negative"
        else:
            continue  # keep only sentiment-bearing opinion words
        out.append({"target": target, "opinion": opinion, "polarity": polarity})
    return out


def run_opinion_extraction(df, sample=1500):
    """Collect opinion triples across a sample and report the most common ones."""
    rows = []
    sub = df.sample(min(sample, len(df)), random_state=config.RANDOM_STATE)
    for _, r in sub.iterrows():
        for t in extract_opinions(r["parse_text"]):
            t["tweet_label"] = r["label"]
            rows.append(t)
    triples = pd.DataFrame(rows)
    triples.to_csv(config.RES_DIR / "opinion_triples.csv", index=False)
    top = (triples.groupby(["target", "opinion", "polarity"])
                  .size().reset_index(name="count")
                  .sort_values("count", ascending=False).head(15))
    print(f"\nExtracted {len(triples)} sentiment-bearing opinion triples "
          f"from {len(sub)} tweets.")
    print("Top target–opinion pairs:")
    print(top.to_string(index=False))
    return triples


def run_opinion_mining(df=None):
    if df is None:
        df = build_clean_dataframe()
    vader = run_vader_baseline(df)
    triples = run_opinion_extraction(df)
    with open(config.RES_DIR / "vader_baseline.json", "w") as fh:
        json.dump(vader, fh, indent=2)
    return vader, triples


if __name__ == "__main__":
    run_opinion_mining()
