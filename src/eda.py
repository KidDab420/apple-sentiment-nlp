"""
eda.py
Exploratory data analysis and visualisation.
Produces report-ready figures in outputs/figures/.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

import config
from data_prep import build_clean_dataframe

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 11


def _palette(labels):
    return [config.SENT_COLORS[l] for l in labels]


def plot_class_distribution(df):
    counts = df["label"].value_counts().reindex(config.LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=_palette(counts.index))
    ax.set_title("Sentiment Class Distribution (Apple Tweets)", fontweight="bold")
    ax.set_ylabel("Number of tweets")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, str(v),
                ha="center", fontweight="bold")
    total = counts.sum()
    ax.text(0.99, 0.95, f"n = {total}", transform=ax.transAxes, ha="right",
            va="top", fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "01_class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_confidence(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl in config.LABEL_ORDER:
        sns.kdeplot(df[df.label == lbl]["confidence"], ax=ax,
                    label=lbl, color=config.SENT_COLORS[lbl], fill=True, alpha=0.25)
    ax.set_title("Annotator Confidence by Sentiment", fontweight="bold")
    ax.set_xlabel("sentiment:confidence")
    ax.legend(title="Sentiment")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "02_confidence_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_tweet_length(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="label", y="word_len", order=config.LABEL_ORDER,
                palette=config.SENT_COLORS, ax=ax)
    ax.set_title("Cleaned Tweet Length (words) by Sentiment", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Word count")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "03_tweet_length.png", bbox_inches="tight")
    plt.close(fig)


def plot_wordclouds(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, lbl in zip(axes, config.LABEL_ORDER):
        text = " ".join(df[df.label == lbl]["clean_text"])
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap="viridis", max_words=80).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{lbl.capitalize()} tweets", fontweight="bold",
                     color=config.SENT_COLORS[lbl])
        ax.axis("off")
    fig.suptitle("Word Clouds by Sentiment", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "04_wordclouds.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_terms(df, n=12):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, lbl in zip(axes, config.LABEL_ORDER):
        words = " ".join(df[df.label == lbl]["clean_text"]).split()
        common = Counter(words).most_common(n)
        terms, freqs = zip(*common)
        ax.barh(range(len(terms)), freqs, color=config.SENT_COLORS[lbl])
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.invert_yaxis()
        ax.set_title(f"Top terms — {lbl}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "05_top_terms.png", bbox_inches="tight")
    plt.close(fig)


def run_eda():
    df = build_clean_dataframe()
    plot_class_distribution(df)
    plot_confidence(df)
    plot_tweet_length(df)
    plot_wordclouds(df)
    plot_top_terms(df)
    print("EDA figures written to", config.FIG_DIR)
    return df


if __name__ == "__main__":
    run_eda()
