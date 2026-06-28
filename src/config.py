"""
config.py
Central configuration for the Apple Twitter Sentiment Analysis pipeline.
CDS6344 - Social Media Computing.
"""
from pathlib import Path

# ----- Paths -----
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

RAW_CSV = DATA_DIR / "Apple-Twitter-Sentiment-DFE.csv"
CLEAN_CSV = DATA_DIR / "apple_sentiment_clean.csv"

for d in (FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----- Label mapping -----
# Original CrowdFlower codes: 1 = negative, 3 = neutral, 5 = positive
LABEL_MAP = {"1": "negative", "3": "neutral", "5": "positive"}
LABEL_ORDER = ["negative", "neutral", "positive"]

# ----- Reproducibility -----
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----- Colour palette (consistent across all figures) -----
SENT_COLORS = {"negative": "#d1495b", "neutral": "#8d99ae", "positive": "#2a9d8f"}
