"""
Pelabelan sentimen (dijalankan SEKALI, hasilnya dibaca oleh aplikasi Streamlit).

Metode sama dengan notebook: model pre-trained
`mdhugol/indonesia-bert-sentiment-classification` pada kolom `clean_text`.

Output:
  data/data_sentiment.csv     -> data_clean + kolom sentiment, sentiment_score
  data/evaluasi_sentiment.csv -> 20 sampel acak (random_state=42) + label manual + prediksi sistem

Cara pakai:
  pip install -r requirements-label.txt
  python label_sentiment.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
IN_PATH = BASE / "data" / "data_clean.csv"
OUT_SENTIMENT = BASE / "data" / "data_sentiment.csv"
OUT_EVAL = BASE / "data" / "evaluasi_sentiment.csv"

PRETRAINED = "mdhugol/indonesia-bert-sentiment-classification"
LABEL_INDEX = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}

# Label manual untuk 20 sampel (df.sample(n=20, random_state=42)), dari notebook.
LABEL_MANUAL = [
    "positive", "negative", "neutral", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "neutral", "positive", "negative", "negative", "positive",
    "neutral", "neutral", "negative", "negative", "negative",
]


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    df = pd.read_csv(IN_PATH)
    print(f"Memuat {len(df)} postingan dari {IN_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED)
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    hasil = clf(df["clean_text"].tolist(), batch_size=32, truncation=True, max_length=512)
    df["sentiment"] = [LABEL_INDEX[h["label"]] for h in hasil]
    df["sentiment_score"] = [h["score"] for h in hasil]
    df.to_csv(OUT_SENTIMENT, index=False)
    print(f"Tersimpan: {OUT_SENTIMENT}")
    print(df["sentiment"].value_counts())

    # Evaluasi manual 20 sampel
    sampel = df.sample(n=20, random_state=42)[["text", "sentiment"]].copy()
    sampel["label_manual"] = LABEL_MANUAL
    sampel["isCocok"] = (sampel["sentiment"] == sampel["label_manual"]).map({True: "Ya", False: "Tidak"})
    sampel.to_csv(OUT_EVAL, index=False)
    print(f"Tersimpan: {OUT_EVAL} (kecocokan {(sampel['isCocok'] == 'Ya').mean():.0%})")


if __name__ == "__main__":
    main()
