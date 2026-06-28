"""
data_prep.py
Loading, cleaning and text pre-processing for the Apple tweet dataset.

Pipeline stages implemented here:
  1. Load raw CSV (latin-1 encoding, mixed line endings).
  2. Drop 'not_relevant' rows and map sentiment codes to labels.
  3. Tweet-specific normalisation (URLs, @mentions, #hashtags, RT, etc.).
  4. Tokenisation, stop-word removal and lemmatisation.
"""
import re
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

# Ensure NLTK resources are present (no-op if already downloaded)
for _r in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(_r)
    except LookupError:
        nltk.download(_r, quiet=True)

LEMMATIZER = WordNetLemmatizer()
# Keep negation words - they carry sentiment signal
_BASE_STOP = set(stopwords.words("english"))
_KEEP = {"not", "no", "nor", "never", "n't", "against"}
STOP_WORDS = _BASE_STOP - _KEEP

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def load_raw() -> pd.DataFrame:
    """Read the raw CSV and standardise the sentiment label."""
    df = pd.read_csv(config.RAW_CSV, encoding="latin-1")
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["sentiment"].isin(config.LABEL_MAP.keys())].copy()
    df["label"] = df["sentiment"].map(config.LABEL_MAP)
    df["confidence"] = pd.to_numeric(df["sentiment:confidence"], errors="coerce")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df


def clean_tweet(text: str, lemmatize: bool = True) -> str:
    """Normalise a single tweet into a clean bag of lemmas."""
    text = html.unescape(str(text))
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)        # '#apple' -> 'apple'
    text = re.sub(r"\brt\b", " ", text)        # retweet marker
    text = re.sub(r"[''`]", "", text)          # keep contractions joined
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in STOP_WORDS]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def clean_for_parse(text: str) -> str:
    """Light cleaning that KEEPS casing/punctuation/grammar for dependency parsing.
    Removes only URLs, @mentions and the '#' symbol (hashtag word is kept)."""
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\bRT\b", " ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def build_clean_dataframe(save: bool = True) -> pd.DataFrame:
    """Full cleaning pass; returns dataframe with a 'clean_text' column."""
    df = load_raw()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df["parse_text"] = df["text"].apply(clean_for_parse)
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["clean_text"].str.split().apply(len)
    # Drop rows that became empty after cleaning
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    if save:
        cols = ["id", "date", "text", "clean_text", "label",
                "confidence", "char_len", "word_len"]
        df[cols].to_csv(config.CLEAN_CSV, index=False)
    return df


if __name__ == "__main__":
    d = build_clean_dataframe()
    print(f"Rows after cleaning: {len(d)}")
    print(d["label"].value_counts())
    print(d[["text", "clean_text", "label"]].head(3).to_string())
