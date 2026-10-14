"""
Clustering teks (dijalankan SEKALI, hasilnya dibaca oleh aplikasi Streamlit).

Metode sama dengan notebook: pembersihan + stemming Sastrawi -> TF-IDF -> K-Means (3 cluster).

Output:
  data/hasil_clustering.csv -> kolom data asli + clean_cluster, n_token, cluster, tema

Cara pakai:
  pip install -r requirements-label.txt
  python cluster_teks.py
"""
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).parent
IN_PATH = BASE / "data" / "data_clean.csv"
OUT_PATH = BASE / "data" / "hasil_clustering.csv"

K = 3
MIN_TOKEN = 3

# Tema hasil interpretasi di notebook (nomor cluster dari K-Means random_state=42).
TOP5_NOTEBOOK = {
    0: ["tugas", "dosen", "bikin", "kuliah", "mahasiswa"],
    1: ["nugas", "kuliah", "ajar", "chatgpt", "kerja"],
    2: ["gue", "ngerjain", "lu", "tugas", "nugas"],
}
LABEL_TEMA = {
    0: "Mahasiswa, Dosen, dan Tugas Kuliah di Era AI",
    1: "Nugas dengan ChatGPT untuk Belajar dan Kerja",
    2: "Pengalaman Pribadi Mengerjakan Tugas dengan AI",
}

nltk.download("stopwords", quiet=True)
stemmer = StemmerFactory().create_stemmer()

# Kamus slang: menyatukan variasi tulisan agar dihitung sebagai satu kata
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak", "gk": "tidak",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "krn": "karena",
    "karna": "karena", "jd": "jadi", "sdh": "sudah", "udh": "sudah", "udah": "sudah",
    "blm": "belum", "tp": "tetapi", "tapi": "tetapi", "bgt": "banget", "aja": "saja",
    "sm": "sama", "org": "orang", "dr": "dari", "dlm": "dalam", "pd": "pada",
    "klo": "kalau", "kalo": "kalau", "jgn": "jangan", "gmn": "bagaimana", "lg": "lagi",
    "skrg": "sekarang", "bkn": "bukan", "dpt": "dapat", "bs": "bisa", "trs": "terus",
    "gw": "gue", "gua": "gue", "lo": "lu", "loe": "lu",
    "pake": "pakai",
}

# Stopword Indonesia + Inggris (tweet sering bercampur dua bahasa) + stopword domain
STOPWORDS = set(nltk_stopwords.words("indonesian")) | set(nltk_stopwords.words("english"))
STOPWORDS |= {
    "ai", "rt", "via", "amp", "dll", "guys",
    "nih", "sih", "deh", "dong", "lah", "kok", "tuh", "dah", "yuk", "ya", "yaa",
    "wkwk", "wkwkwk", "haha", "banget", "nya", "pas", "biar", "gitu", "kayak",
    "gimana", "sampe", "tau", "liat", "orang", "pakai",
}


@lru_cache(maxsize=None)
def stem_word(word: str) -> str:
    return stemmer.stem(word)  # cache karena stemming Sastrawi lambat


def bersihkan_teks(text, stem=True, min_len=2) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"@\w+", " ", text)                    # mention
    text = re.sub(r"^rt\s+", " ", text)                  # penanda retweet
    text = re.sub(r"#(\w+)", r" \1 ", text)              # hashtag: simbol dibuang, kata dipertahankan
    text = re.sub(r"[^a-z\s]", " ", text)                # selain huruf
    text = re.sub(r"(.)\1{2,}", r"\1", text)             # huruf berulang >= 3
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for tok in text.split():
        tok = SLANG.get(tok, tok)
        if tok in STOPWORDS or len(tok) < min_len:
            continue
        if stem:
            tok = stem_word(tok)
        if tok not in STOPWORDS and len(tok) >= min_len:
            tokens.append(tok)
    return " ".join(tokens)


def main() -> None:
    with open(IN_PATH, encoding="utf-8-sig") as f:
        header = f.readline()
    sep = max((",", "|", ";", "\t"), key=header.count)  # notebook menyimpan CSV dengan pemisah '|'
    data = pd.read_csv(IN_PATH, sep=sep)
    assert "text" in data.columns, "Kolom 'text' tidak ada"
    print(f"Memuat {len(data)} postingan dari {IN_PATH}")

    df = data.copy()
    df["clean_cluster"] = df["text"].apply(bersihkan_teks)
    df["n_token"] = df["clean_cluster"].str.split().str.len()
    df = df[df["n_token"] >= MIN_TOKEN].reset_index(drop=True)
    print(f"Dipakai: {len(df)} | Dikeluarkan: {len(data) - len(df)} (kurang dari {MIN_TOKEN} kata)")

    vec = TfidfVectorizer(max_df=0.8, min_df=2)
    X = vec.fit_transform(df["clean_cluster"])
    print("Ukuran matriks:", X.shape)

    model = KMeans(n_clusters=K, init="k-means++", max_iter=100, n_init=10, random_state=42)
    model.fit(X)
    df["cluster"] = model.labels_
    print(df["cluster"].value_counts().sort_index().to_string())

    terms = vec.get_feature_names_out()
    urut = model.cluster_centers_.argsort()[:, ::-1]
    top5 = {i: [terms[j] for j in urut[i, :5]] for i in range(K)}
    for i in range(K):
        print(f"Cluster {i}: {', '.join(top5[i])}")

    # Tema dari notebook hanya dipasang bila kata dominannya cocok; bila tidak, beri label netral.
    cocok = all(Counter(top5[i]) == Counter(TOP5_NOTEBOOK[i]) for i in range(K))
    if cocok:
        df["tema"] = df["cluster"].map(lambda i: f"{LABEL_TEMA[i]} ({', '.join(TOP5_NOTEBOOK[i])})")
    else:
        print("PERINGATAN: kata dominan berbeda dari notebook; tema diberi label sementara. "
              "Perbarui TOP5_NOTEBOOK dan LABEL_TEMA setelah meninjau hasil.")
        df["tema"] = df["cluster"].map(lambda i: f"Cluster {i} ({', '.join(top5[i])})")

    df.to_csv(OUT_PATH, index=False)
    print(f"Tersimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
