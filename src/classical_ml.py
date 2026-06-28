"""
classical_ml.py
Traditional machine-learning sentiment classifiers on TF-IDF features.

Models: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
Evaluation: accuracy, macro precision/recall/F1, per-class report,
confusion matrices, 5-fold cross-validation, model comparison chart.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import config
from data_prep import build_clean_dataframe

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130


def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1),
    }


def plot_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.LABEL_ORDER, yticklabels=config.LABEL_ORDER, ax=ax)
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fname = name.lower().replace(" ", "_")
    fig.savefig(config.FIG_DIR / f"06_cm_{fname}.png", bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    melt = results_df.melt(id_vars="Model",
                           value_vars=["Accuracy", "Macro F1"],
                           var_name="Metric", value_name="Score")
    sns.barplot(data=melt, x="Model", y="Score", hue="Metric", ax=ax,
                palette=["#264653", "#e76f51"])
    ax.set_title("Classical ML Model Comparison", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "07_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def run_classical(df=None):
    if df is None:
        df = build_clean_dataframe()

    X, y = df["clean_text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=8000,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    print(f"TF-IDF matrix: {Xtr.shape[0]} train x {Xtr.shape[1]} features")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rows, reports = [], {}

    for name, model in get_models().items():
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)
        acc = accuracy_score(y_te, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="macro", zero_division=0)
        cv_f1 = cross_val_score(model, vec.transform(X), y, cv=cv,
                                scoring="f1_macro", n_jobs=-1)
        rows.append({"Model": name, "Accuracy": round(acc, 4),
                     "Macro Precision": round(p, 4), "Macro Recall": round(r, 4),
                     "Macro F1": round(f1, 4),
                     "CV F1 (mean)": round(cv_f1.mean(), 4),
                     "CV F1 (std)": round(cv_f1.std(), 4)})
        reports[name] = classification_report(
            y_te, pred, target_names=config.LABEL_ORDER, zero_division=0)
        plot_confusion(confusion_matrix(y_te, pred, labels=config.LABEL_ORDER), name)
        print(f"  {name:<22} acc={acc:.3f}  macroF1={f1:.3f}  cvF1={cv_f1.mean():.3f}")

    results = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    results.to_csv(config.RES_DIR / "classical_metrics.csv", index=False)
    plot_comparison(results)

    with open(config.RES_DIR / "classification_reports.txt", "w") as fh:
        for n, rep in reports.items():
            fh.write(f"===== {n} =====\n{rep}\n\n")

    print("\nLeaderboard (by Macro F1):")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_classical()
