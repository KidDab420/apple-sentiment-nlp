"""
run_pipeline.py
End-to-end runner for the classical / lexicon / opinion / ABSA stages.
(Deep-learning & Transformer models live in the Colab notebook, which needs a GPU.)

Usage:  python run_pipeline.py
"""
import warnings
warnings.filterwarnings("ignore")
import json
import time

import config
from data_prep import build_clean_dataframe
from eda import (plot_class_distribution, plot_confidence, plot_tweet_length,
                 plot_wordclouds, plot_top_terms)
from classical_ml import run_classical
from opinion_mining import run_opinion_mining
from absa import run_absa


def main():
    t0 = time.time()
    print("=" * 60)
    print(" APPLE TWITTER SENTIMENT — NLP PIPELINE (CDS6344)")
    print("=" * 60)

    print("\n[1/5] Cleaning & preprocessing ...")
    df = build_clean_dataframe()
    print(f"      {len(df)} tweets | classes: {df['label'].value_counts().to_dict()}")

    print("\n[2/5] EDA & visualisation ...")
    plot_class_distribution(df); plot_confidence(df); plot_tweet_length(df)
    plot_wordclouds(df); plot_top_terms(df)

    print("\n[3/5] Classical ML models ...")
    clf_results = run_classical(df)

    print("\n[4/5] Opinion mining (VADER + dependency parsing) ...")
    vader, _ = run_opinion_mining(df)

    print("\n[5/5] Aspect-Based Sentiment Analysis ...")
    absa_df = run_absa(df)

    summary = {
        "n_tweets": int(len(df)),
        "class_distribution": df["label"].value_counts().to_dict(),
        "best_classical_model": clf_results.iloc[0]["Model"],
        "best_macro_f1": float(clf_results.iloc[0]["Macro F1"]),
        "vader_baseline": vader,
        "n_aspects": int(len(absa_df)),
        "most_negative_aspect": absa_df.sort_values("net_sentiment").iloc[0]["aspect"],
        "most_positive_aspect": absa_df.sort_values("net_sentiment").iloc[-1]["aspect"],
    }
    with open(config.RES_DIR / "pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 60)
    print(f" DONE in {time.time()-t0:.1f}s. Figures -> {config.FIG_DIR}")
    print(f" Results -> {config.RES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
