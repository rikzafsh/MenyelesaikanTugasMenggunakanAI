"""
Dashboard Analisis Sentimen dan Tema Pembahasan Penggunaan AI untuk Membuat
dan Menyelesaikan Tugas pada Postingan X Berbahasa Indonesia.

Tab: Data, Sentimen, Clustering, dan Regresi.

Jalankan:  streamlit run app.py
"""
import warnings
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
BASE = Path(__file__).parent
PATH_CLEAN = BASE / "data" / "data_clean.csv"
PATH_SENTIMENT = BASE / "data" / "data_sentiment.csv"
PATH_EVAL = BASE / "data" / "evaluasi_sentiment.csv"
PATH_CLUSTER = BASE / "data" / "hasil_clustering.csv"
PATH_MODEL = BASE / "model" / "model_regresi_ai.joblib"

# Warna cluster dibuat berbeda dari warna sentimen agar tidak tertukar.
WARNA_CLUSTER = ["#4C78A8", "#F58518", "#B279A2", "#72B7B2", "#E45756"]

# Regresi: X1 = reply_count, X2 = retweet_count, Y = like_count (sesuai notebook)
FITUR_REGRESI = ["reply_count", "retweet_count"]
TARGET_REGRESI = "like_count"
LABEL_FITUR = {"reply_count": "Jumlah reply", "retweet_count": "Jumlah retweet",
               "like_count": "Jumlah likes"}

# Hasil dari notebook (dipakai hanya bila dataset belum memuat reply_count/retweet_count).
HASIL_NOTEBOOK = {
    "mae": 81.19,
    "r2": 0.5916,
    "korelasi": {("like_count", "reply_count"): 0.91,
                 ("like_count", "retweet_count"): 0.98,
                 ("reply_count", "retweet_count"): 0.88},
}

URUTAN_SENTIMEN = ["positive", "neutral", "negative"]
LABEL_ID = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}
WARNA = {"Positif": "#2E8B57", "Netral": "#8A8F98", "Negatif": "#D1495B"}
URUTAN_ID = [LABEL_ID[s] for s in URUTAN_SENTIMEN]
SKALA_WARNA = alt.Scale(domain=URUTAN_ID, range=[WARNA[k] for k in URUTAN_ID])

# Stopword ringan (bahasa Indonesia + bahasa gaul) untuk daftar kata teratas.
STOPWORDS = set("""
yang dan di ke dari ini itu untuk dengan pada dalam atau juga karena tapi tetapi
jadi kalau kalo kalian kita kami saya aku kamu dia mereka nya nih deh sih dong lah
kok aja saja udah sudah masih lagi bisa buat bikin banget bgt gak ga nggak tidak
bukan ada apa siapa kenapa gimana bagaimana yg dgn utk dr krn tp jd klo sama lebih
sangat sekali pake pakai pakek pas lalu terus dulu yaa ya iya
""".split())


# --------------------------------------------------------------------------
# Pemuat data
# --------------------------------------------------------------------------
def _baca_csv(path: Path) -> pd.DataFrame:
    """Baca CSV dengan pemisah otomatis (koma atau '|', format simpanan notebook)."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    sep = max((",", "|", ";", "\t"), key=header.count)
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner=False)
def muat_data(path: str) -> pd.DataFrame:
    df = _baca_csv(Path(path))
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["tahun"] = df["created_at"].dt.year
    df["jumlah_kata"] = df["clean_text"].fillna("").str.split().str.len()
    return df


@st.cache_data(show_spinner=False)
def muat_sentimen(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = _baca_csv(p)
    if "sentiment" not in df.columns:
        return None
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["tahun"] = df["created_at"].dt.year
    df["jumlah_kata"] = df["clean_text"].fillna("").str.split().str.len()
    df["sentimen"] = df["sentiment"].map(LABEL_ID)
    return df


@st.cache_data(show_spinner=False)
def muat_evaluasi(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = _baca_csv(p)
    if not {"sentiment", "label_manual"}.issubset(df.columns):
        return None
    return df


def kata_teratas(teks: pd.Series, n: int = 15) -> pd.DataFrame:
    hitung = Counter()
    for t in teks.dropna():
        hitung.update(w for w in t.split() if len(w) > 2 and w not in STOPWORDS)
    return pd.DataFrame(hitung.most_common(n), columns=["kata", "frekuensi"])


def grafik_batang_kata(df_kata: pd.DataFrame, warna: str) -> alt.Chart:
    return (
        alt.Chart(df_kata)
        .mark_bar(color=warna)
        .encode(
            x=alt.X("frekuensi:Q", title="Frekuensi"),
            y=alt.Y("kata:N", sort="-x", title=None),
            tooltip=["kata", "frekuensi"],
        )
        .properties(height=alt.Step(22))
    )


@st.cache_data(show_spinner="Menghitung ruang fitur cluster...")
def analisis_cluster(path: str):
    """Muat hasil clustering lalu hitung kata dominan, silhouette, dan proyeksi PCA.

    Hanya membutuhkan kolom `text`, `clean_cluster`, `cluster` (opsional: `tema`).
    TF-IDF memakai pengaturan yang sama dengan notebook (max_df=0.8, min_df=2).
    """
    p = Path(path)
    if not p.exists():
        return None
    d = _baca_csv(p)
    if not {"text", "clean_cluster", "cluster"}.issubset(d.columns):
        return None
    d = d.dropna(subset=["clean_cluster"]).reset_index(drop=True)
    d["cluster"] = d["cluster"].astype(int)

    vec = TfidfVectorizer(max_df=0.8, min_df=2)
    X = vec.fit_transform(d["clean_cluster"]).toarray()
    labels = d["cluster"].to_numpy()
    ids = sorted(d["cluster"].unique().tolist())
    istilah = vec.get_feature_names_out()

    pusat = np.vstack([X[labels == i].mean(axis=0) for i in ids])   # pusat cluster = rata-rata TF-IDF
    posisi = {i: k for k, i in enumerate(ids)}
    d["jarak"] = np.linalg.norm(X - pusat[[posisi[i] for i in labels]], axis=1)

    kata = {}
    for i in ids:
        urut = np.argsort(pusat[posisi[i]])[::-1][:10]
        kata[i] = pd.DataFrame({"kata": istilah[urut], "bobot": pusat[posisi[i]][urut]})

    sil, sil_cluster = None, {}
    if len(ids) >= 2 and len(d) > len(ids):
        sil = float(silhouette_score(X, labels))
        sampel = silhouette_samples(X, labels)
        sil_cluster = {i: float(sampel[labels == i].mean()) for i in ids}

    pca = PCA(n_components=2, random_state=0).fit(X)
    xy = pca.transform(X)
    d["pca1"], d["pca2"] = xy[:, 0], xy[:, 1]
    xy_pusat = pca.transform(pusat)
    titik_pusat = pd.DataFrame({"cluster": [f"Cluster {i}" for i in ids],
                                "pca1": xy_pusat[:, 0], "pca2": xy_pusat[:, 1]})

    if "tema" in d.columns:
        d["tema_pendek"] = d["tema"].astype(str).str.replace(r"\s*\(.*\)\s*$", "", regex=True)
    else:
        d["tema_pendek"] = "Cluster " + d["cluster"].astype(str)
    d["nama"] = "Cluster " + d["cluster"].astype(str)

    return {"data": d, "ids": ids, "kata": kata, "silhouette": sil, "sil_cluster": sil_cluster,
            "titik_pusat": titik_pusat, "var_pca": float(pca.explained_variance_ratio_.sum()),
            "n_fitur": int(X.shape[1])}


@st.cache_data(show_spinner=False)
def buat_wordcloud(teks: str, colormap: str = "viridis"):
    """Word cloud dari teks bersih (ukuran kata = frekuensi). Mengembalikan (gambar, 5 kata teratas)."""
    if not teks.strip():
        return None, []
    wc = WordCloud(stopwords=set(), background_color="black", max_words=500, width=800, height=400,
                   collocations=False, colormap=colormap, random_state=42).generate(teks)
    return wc.to_array(), list(wc.words_.keys())[:5]


def grafik_bobot_kata(df_kata: pd.DataFrame, warna: str) -> alt.Chart:
    return (
        alt.Chart(df_kata)
        .mark_bar(color=warna)
        .encode(
            x=alt.X("bobot:Q", title="Bobot TF-IDF (pusat cluster)"),
            y=alt.Y("kata:N", sort="-x", title=None),
            tooltip=["kata", alt.Tooltip("bobot:Q", format=".3f")],
        )
        .properties(height=alt.Step(22))
    )


@st.cache_resource(show_spinner=False)
def muat_model(path: str):
    """Muat model regresi tersimpan (joblib). Mengembalikan None bila gagal."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # model disimpan dengan versi scikit-learn lain
            model = joblib.load(p)
        if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
            return None
        return model
    except Exception:
        return None


def prediksi_manual(model, nilai: list[float]) -> float:
    """Regresi linear = intersep + jumlah(koefisien x nilai)."""
    return float(model.intercept_ + np.dot(np.ravel(model.coef_), nilai))


def matriks_korelasi_notebook() -> pd.DataFrame:
    nama = ["like_count", "reply_count", "retweet_count"]
    m = pd.DataFrame(np.eye(3), index=nama, columns=nama)
    for (a, b), r in HASIL_NOTEBOOK["korelasi"].items():
        m.loc[a, b] = m.loc[b, a] = r
    return m


def grafik_korelasi(korelasi: pd.DataFrame) -> alt.Chart:
    panjang = (korelasi.rename_axis("x").reset_index()
               .melt(id_vars="x", var_name="y", value_name="r"))
    dasar = alt.Chart(panjang).encode(x=alt.X("x:N", title=None, axis=alt.Axis(labelAngle=0)),
                                      y=alt.Y("y:N", title=None))
    kotak = dasar.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True),
                        legend=alt.Legend(title="Korelasi")),
        tooltip=["x", "y", alt.Tooltip("r:Q", format=".2f")],
    )
    angka = dasar.mark_text(fontSize=14).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.5", alt.value("white"), alt.value("black")),
    )
    return (kotak + angka).properties(height=260)


def evaluasi_regresi(data: pd.DataFrame, model):
    """Bagi data 80:20 (random_state=42, sama dengan notebook) lalu evaluasi model tersimpan."""
    d = data.dropna(subset=FITUR_REGRESI + [TARGET_REGRESI])
    X, y = d[FITUR_REGRESI], d[TARGET_REGRESI]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = X_te.to_numpy() @ np.ravel(model.coef_) + float(model.intercept_)
    # Cek: apakah pelatihan ulang pada data ini menghasilkan koefisien yang sama dengan model tersimpan?
    ulang = LinearRegression().fit(X_tr, y_tr)
    sama = bool(np.allclose(ulang.coef_, model.coef_, rtol=1e-4)
                and np.isclose(ulang.intercept_, model.intercept_, rtol=1e-4))
    hasil = X_te.copy()
    hasil["aktual"], hasil["prediksi"] = y_te, pred
    hasil["selisih"] = hasil["prediksi"] - hasil["aktual"]
    hasil["text"] = d.loc[hasil.index, "text"]
    return {"mae": mean_absolute_error(y_te, pred), "r2": r2_score(y_te, pred),
            "n_latih": len(X_tr), "n_uji": len(X_te), "sama": sama,
            "hasil": hasil, "korelasi": d[[TARGET_REGRESI] + FITUR_REGRESI].corr()}


# --------------------------------------------------------------------------
# Halaman
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Sentimen: AI untuk Tugas Kuliah",
    page_icon="📊",
    layout="wide",
)

st.title("Dashboard Analisis Sentimen dan Tema Pembahasan")
st.caption(
    "Penggunaan AI untuk membuat dan menyelesaikan tugas kuliah pada postingan X "
    "berbahasa Indonesia."
)

if not PATH_CLEAN.exists():
    st.error(f"File data tidak ditemukan: `{PATH_CLEAN.relative_to(BASE)}`. "
             "Letakkan dataset bersih di folder `data/`.")
    st.stop()

df = muat_data(str(PATH_CLEAN))
df_sent = muat_sentimen(str(PATH_SENTIMENT))
hasil_cluster = analisis_cluster(str(PATH_CLUSTER))
model_reg = muat_model(str(PATH_MODEL))
ada_data_regresi = set(FITUR_REGRESI + [TARGET_REGRESI]).issubset(df.columns)

with st.sidebar:
    st.header("Tentang dashboard")
    st.write(
        "Dashboard ini membaca dataset CSV yang sudah disiapkan sebelumnya "
        "(tidak ada pengambilan data real-time)."
    )
    st.markdown("**Status berkas**")
    st.markdown(f"- Data bersih: {'✅' if PATH_CLEAN.exists() else '❌'}")
    st.markdown(f"- Hasil sentimen: {'✅' if df_sent is not None else '❌ belum ada'}")
    st.markdown(f"- Evaluasi sentimen: {'✅' if PATH_EVAL.exists() else '❌ belum ada'}")
    st.markdown(f"- Hasil clustering: {'✅' if hasil_cluster is not None else '❌ belum ada'}")
    st.markdown(f"- Model regresi: {'✅' if model_reg is not None else '❌ belum ada'}")
    st.markdown(f"- Data reply/retweet: {'✅' if ada_data_regresi else '❌ belum ada'}")

tab_data, tab_sentimen, tab_cluster, tab_regresi = st.tabs(["Data", "Sentimen", "Clustering", "Regresi"])

# ==========================================================================
# TAB DATA
# ==========================================================================
with tab_data:
    st.subheader("Ringkasan dataset")

    tgl_min, tgl_max = df["created_at"].min(), df["created_at"].max()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah postingan", f"{len(df):,}".replace(",", "."))
    k2.metric("Rentang waktu", f"{tgl_min:%Y} – {tgl_max:%Y}")
    k3.metric("Median likes", f"{df['like_count'].median():.0f}")
    k4.metric("Postingan ber-hashtag", f"{(df['hashtag_count'] > 0).sum()}")

    if len(df) >= 150:
        st.caption(f"Memenuhi target minimal 150 postingan unik "
                   f"({df['clean_text'].nunique()} teks bersih unik).")

    with st.expander("Alur pembersihan data"):
        st.markdown(
            """
            Data mentah digabung dari beberapa kata kunci pencarian, lalu dibersihkan
            dengan langkah berikut:

            1. Hapus teks kosong dan postingan duplikat.
            2. Pastikan jumlah likes berupa angka valid (tidak negatif).
            3. Hitung jumlah hashtag dari teks asli (sebelum simbol `#` dihapus).
            4. Bersihkan teks dengan regex: huruf kecil, hapus tautan, *mention*, angka,
               tanda baca, dan emoji. **Kata negasi (tidak, bukan, dll.) dipertahankan.**
            5. Buang postingan promosi joki / jasa tugas.
            6. Buang postingan yang tidak memuat kata akademik **dan** kata AI.
            7. Buang teks yang terlalu pendek (kurang dari 3 kata) dan duplikat teks bersih.

            Hasil: **588 → 192 postingan**
            (promosi joki 43 baris; tidak sesuai topik 275 baris).
            """
        )

    st.divider()
    st.subheader("Jelajahi data")

    f1, f2, f3 = st.columns([3, 1, 1])
    kata_cari = f1.text_input("Cari kata dalam postingan", placeholder="mis. dosen, chatgpt, plagiat")
    hanya_hashtag = f2.checkbox("Hanya ber-hashtag")
    urut = f3.selectbox("Urutkan", ["Terbaru", "Likes tertinggi", "Terlama"])

    tampil = df.copy()
    if kata_cari:
        tampil = tampil[tampil["text"].str.contains(kata_cari, case=False, na=False, regex=False)]
    if hanya_hashtag:
        tampil = tampil[tampil["hashtag_count"] > 0]
    if urut == "Terbaru":
        tampil = tampil.sort_values("created_at", ascending=False)
    elif urut == "Terlama":
        tampil = tampil.sort_values("created_at")
    else:
        tampil = tampil.sort_values("like_count", ascending=False)

    st.caption(f"Menampilkan {len(tampil)} dari {len(df)} postingan.")
    st.dataframe(
        tampil[["text", "like_count", "hashtag_count", "created_at"]],
        width="stretch",
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn("Postingan", width="large"),
            "like_count": st.column_config.NumberColumn("Likes"),
            "hashtag_count": st.column_config.NumberColumn("Hashtag"),
            "created_at": st.column_config.DatetimeColumn("Waktu", format="DD MMM YYYY"),
        },
    )
    st.download_button(
        "Unduh data bersih (CSV)",
        df.drop(columns=["tahun", "jumlah_kata"]).to_csv(index=False).encode("utf-8"),
        file_name="data_clean.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Gambaran umum data")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Jumlah postingan per tahun**")
        per_tahun = df.groupby("tahun").size().reset_index(name="jumlah")
        st.altair_chart(
            alt.Chart(per_tahun).mark_bar(color="#4C78A8").encode(
                x=alt.X("tahun:O", title="Tahun"),
                y=alt.Y("jumlah:Q", title="Jumlah postingan"),
                tooltip=["tahun", "jumlah"],
            ),
            width="stretch",
        )
    with g2:
        st.markdown("**Distribusi jumlah kata (teks bersih)**")
        st.altair_chart(
            alt.Chart(df).mark_bar(color="#4C78A8").encode(
                x=alt.X("jumlah_kata:Q", bin=alt.Bin(maxbins=15), title="Jumlah kata"),
                y=alt.Y("count()", title="Jumlah postingan"),
            ),
            width="stretch",
        )

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Sebaran jumlah likes**")
        batas = [-1, 0, 10, 100, 1000, float("inf")]
        nama = ["0", "1–10", "11–100", "101–1.000", "> 1.000"]
        kelompok = pd.cut(df["like_count"], bins=batas, labels=nama).value_counts().reindex(nama)
        df_like = kelompok.rename_axis("likes").reset_index(name="jumlah")
        st.altair_chart(
            alt.Chart(df_like).mark_bar(color="#4C78A8").encode(
                x=alt.X("likes:N", sort=nama, title="Kelompok likes"),
                y=alt.Y("jumlah:Q", title="Jumlah postingan"),
                tooltip=["likes", "jumlah"],
            ),
            width="stretch",
        )
        likes_maks = f"{df['like_count'].max():,}".replace(",", ".")
        st.caption(
            f"Sebaran sangat condong: setengah postingan hanya punya ≤ {df['like_count'].median():.0f} likes, "
            f"sedangkan yang tertinggi mencapai {likes_maks}."
        )
    with g4:
        st.markdown("**Kata yang paling sering muncul**")
        st.altair_chart(grafik_batang_kata(kata_teratas(df["clean_text"]), "#4C78A8"),
                        width="stretch")

# ==========================================================================
# TAB SENTIMEN
# ==========================================================================
with tab_sentimen:
    if df_sent is None:
        st.info(
            "Hasil pelabelan sentimen belum tersedia. Jalankan sekali:\n\n"
            "```bash\npip install -r requirements-label.txt\npython label_sentiment.py\n```\n\n"
            "Skrip akan membuat `data/data_sentiment.csv` dan `data/evaluasi_sentiment.csv`, "
            "lalu muat ulang halaman ini."
        )
    else:
        st.subheader("Distribusi sentimen")
        total = len(df_sent)
        jumlah = df_sent["sentimen"].value_counts().reindex(URUTAN_ID).fillna(0).astype(int)
        persen = (jumlah / total * 100).round(1)

        m1, m2, m3 = st.columns(3)
        for kol, label in zip((m1, m2, m3), URUTAN_ID):
            kol.metric(label, f"{jumlah[label]}", f"{persen[label]}% dari {total}", delta_color="off")

        df_dist = pd.DataFrame({"sentimen": URUTAN_ID, "jumlah": jumlah.values, "persen": persen.values})
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                alt.Chart(df_dist).mark_bar().encode(
                    x=alt.X("sentimen:N", sort=URUTAN_ID, title=None),
                    y=alt.Y("jumlah:Q", title="Jumlah postingan"),
                    color=alt.Color("sentimen:N", scale=SKALA_WARNA, legend=None),
                    tooltip=["sentimen", "jumlah", alt.Tooltip("persen:Q", title="Persen (%)")],
                ),
                width="stretch",
            )
        with c2:
            st.altair_chart(
                alt.Chart(df_dist).mark_arc(innerRadius=60).encode(
                    theta="jumlah:Q",
                    color=alt.Color("sentimen:N", scale=SKALA_WARNA, title="Sentimen"),
                    tooltip=["sentimen", "jumlah", alt.Tooltip("persen:Q", title="Persen (%)")],
                ),
                width="stretch",
            )

        st.divider()
        st.subheader("Sentimen dan keterlibatan")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Rata-rata dan median likes per sentimen**")
            ringkas = (
                df_sent.groupby("sentimen")["like_count"]
                .agg(rata_rata="mean", median="median", maksimum="max")
                .reindex(URUTAN_ID).round(1).reset_index()
                .rename(columns={"sentimen": "Sentimen", "rata_rata": "Rata-rata",
                                 "median": "Median", "maksimum": "Maksimum"})
            )
            st.dataframe(ringkas, hide_index=True, width="stretch")
            st.caption("Rata-rata mudah terdorong oleh satu-dua postingan viral; median lebih mewakili.")
        with s2:
            st.markdown("**Komposisi sentimen per tahun**")
            per_thn = df_sent.groupby(["tahun", "sentimen"]).size().reset_index(name="jumlah")
            st.altair_chart(
                alt.Chart(per_thn).mark_bar().encode(
                    x=alt.X("tahun:O", title="Tahun"),
                    y=alt.Y("jumlah:Q", title="Jumlah postingan", stack="zero"),
                    color=alt.Color("sentimen:N", scale=SKALA_WARNA, title="Sentimen"),
                    tooltip=["tahun", "sentimen", "jumlah"],
                ),
                width="stretch",
            )

        st.divider()
        st.subheader("Kata dominan per sentimen")
        kolom_kata = st.columns(3)
        for kol, label in zip(kolom_kata, URUTAN_ID):
            with kol:
                st.markdown(f"**{label}**")
                subset = df_sent.loc[df_sent["sentimen"] == label, "clean_text"]
                if subset.empty:
                    st.caption("Tidak ada postingan.")
                else:
                    st.altair_chart(grafik_batang_kata(kata_teratas(subset, 10), WARNA[label]),
                                    width="stretch")

        st.divider()
        st.subheader("Contoh postingan per sentimen")
        e1, e2, e3 = st.columns([2, 2, 2])
        pilih = e1.multiselect("Sentimen", URUTAN_ID, default=URUTAN_ID)
        if "sentiment_score" in df_sent.columns:
            min_skor = e2.slider("Keyakinan model minimal", 0.0, 1.0, 0.0, 0.05)
        else:
            min_skor = 0.0
        urut_s = e3.selectbox("Urutkan contoh", ["Likes tertinggi", "Keyakinan tertinggi", "Terbaru"])

        contoh = df_sent[df_sent["sentimen"].isin(pilih)]
        if "sentiment_score" in contoh.columns:
            contoh = contoh[contoh["sentiment_score"] >= min_skor]
        if urut_s == "Likes tertinggi":
            contoh = contoh.sort_values("like_count", ascending=False)
        elif urut_s == "Keyakinan tertinggi" and "sentiment_score" in contoh.columns:
            contoh = contoh.sort_values("sentiment_score", ascending=False)
        else:
            contoh = contoh.sort_values("created_at", ascending=False)

        kolom_tabel = ["text", "sentimen"] + (["sentiment_score"] if "sentiment_score" in contoh.columns else []) \
                      + ["like_count"]
        st.caption(f"{len(contoh)} postingan sesuai filter.")
        st.dataframe(
            contoh[kolom_tabel],
            width="stretch",
            hide_index=True,
            column_config={
                "text": st.column_config.TextColumn("Postingan", width="large"),
                "sentimen": "Sentimen",
                "sentiment_score": st.column_config.ProgressColumn("Keyakinan", min_value=0.0, max_value=1.0,
                                                                    format="%.2f"),
                "like_count": "Likes",
            },
        )

        # ------------------------------------------------------------------
        # Evaluasi manual
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Validasi manual (20 sampel)")
        ev = muat_evaluasi(str(PATH_EVAL))
        if ev is None:
            st.info("Berkas `data/evaluasi_sentiment.csv` belum tersedia "
                    "(kolom yang dibutuhkan: `text`, `sentiment`, `label_manual`).")
        else:
            ev = ev.copy()
            ev["cocok"] = ev["sentiment"] == ev["label_manual"]
            akurasi = ev["cocok"].mean() * 100
            a1, a2 = st.columns([1, 2])
            a1.metric("Kecocokan label sistem vs manual", f"{akurasi:.0f}%",
                      f"{int(ev['cocok'].sum())} dari {len(ev)} sampel", delta_color="off")
            with a2:
                matriks = pd.crosstab(
                    ev["label_manual"].map(LABEL_ID).rename("Label manual"),
                    ev["sentiment"].map(LABEL_ID).rename("Prediksi sistem"),
                ).reindex(index=URUTAN_ID, columns=URUTAN_ID, fill_value=0)
                st.markdown("**Matriks kebingungan**")
                st.dataframe(matriks, width="stretch")

            with st.expander("Lihat 20 sampel dan ketidaksesuaian"):
                tabel_ev = ev.assign(
                    **{"Label manual": ev["label_manual"].map(LABEL_ID),
                       "Prediksi sistem": ev["sentiment"].map(LABEL_ID),
                       "Cocok": ev["cocok"].map({True: "Ya", False: "Tidak"})}
                )[["text", "Label manual", "Prediksi sistem", "Cocok"]]
                st.dataframe(tabel_ev, hide_index=True, width="stretch",
                             column_config={"text": st.column_config.TextColumn("Postingan", width="large")})

        st.caption(
            "Catatan: label sentimen berasal dari model pre-trained, sehingga dapat keliru pada "
            "bahasa gaul, singkatan, sarkasme, dan teks yang memuat beberapa target opini sekaligus."
        )


# ==========================================================================
# TAB CLUSTERING
# ==========================================================================
with tab_cluster:
    if hasil_cluster is None:
        st.info(
            "Hasil clustering belum tersedia. Jalankan sekali:\n\n"
            "```bash\npip install -r requirements-label.txt\npython cluster_teks.py\n```\n\n"
            "Skrip akan membuat `data/hasil_clustering.csv` (kolom `text`, `clean_cluster`, `cluster`, `tema`), "
            "lalu muat ulang halaman ini. File `hasil_clustering.csv` dari notebook juga dapat dipakai langsung."
        )
    else:
        dc = hasil_cluster["data"]
        if "like_count" not in dc.columns:   # berkas clustering minimal: ambil likes dari data utama
            dc["like_count"] = dc["text"].map(df.drop_duplicates("text").set_index("text")["like_count"])
        ids = hasil_cluster["ids"]
        sil = hasil_cluster["silhouette"]
        warna_c = {f"Cluster {i}": WARNA_CLUSTER[k % len(WARNA_CLUSTER)] for k, i in enumerate(ids)}
        skala_c = alt.Scale(domain=list(warna_c), range=list(warna_c.values()))
        if df_sent is not None:
            dc["sentimen"] = dc["text"].map(df_sent.drop_duplicates("text").set_index("text")["sentimen"])

        st.subheader("Kelompok tema pembahasan")
        st.markdown(
            "Postingan dikelompokkan berdasarkan kemiripan kata memakai **TF-IDF** dan **K-Means** "
            f"({len(ids)} cluster). Teks dibersihkan lebih lanjut (normalisasi slang, penghapusan stopword, "
            "stemming) sebelum dikelompokkan."
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Postingan dikelompokkan", f"{len(dc)}")
        k2.metric("Jumlah fitur (kata)", f"{hasil_cluster['n_fitur']}")
        k3.metric("Skor silhouette", f"{sil:.3f}" if sil is not None else "-",
                  help="Mengukur seberapa jelas cluster terpisah: −1 s.d. 1. Makin tinggi makin tegas.")
        if sil is not None and sil < 0.25:
            st.warning(
                f"Skor silhouette {sil:.3f} sangat rendah (di bawah 0,25 umumnya dianggap tidak ada struktur "
                "cluster yang jelas). Cluster saling tumpang tindih, jadi tema di bawah lebih tepat dibaca "
                "sebagai kecenderungan gaya pembahasan, bukan kelompok yang terpisah tegas."
            )

        # ---- ukuran & ciri tiap cluster
        st.divider()
        st.subheader("Ciri utama setiap cluster")
        ukuran = dc["nama"].value_counts().reindex(list(warna_c)).fillna(0).astype(int)
        kolom = st.columns(len(ids))
        for kol, i in zip(kolom, ids):
            sub = dc[dc["cluster"] == i]
            with kol:
                with st.container(border=True):
                    st.markdown(f"**Cluster {i}**")
                    st.write(sub["tema_pendek"].iloc[0])
                    st.metric("Jumlah postingan", f"{len(sub)}",
                              f"{len(sub) / len(dc) * 100:.1f}% dari total", delta_color="off")
                    st.caption(f"Median likes: {sub['like_count'].median():.0f}"
                               + (f" · silhouette: {hasil_cluster['sil_cluster'][i]:.2f}"
                                  if i in hasil_cluster["sil_cluster"] else ""))
                    st.caption("Kata dominan: " + ", ".join(hasil_cluster["kata"][i]["kata"].head(5)))

        top5 = [k for i in ids for k in hasil_cluster["kata"][i]["kata"].head(5)]
        tumpang = sorted(k for k, n in Counter(top5).items() if n > 1)
        if tumpang:
            st.caption("Kata yang muncul di lebih dari satu cluster (lima kata teratas): "
                       + ", ".join(tumpang) + ". Ini salah satu sebab cluster saling tumpang tindih.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Jumlah postingan per cluster**")
            df_uk = pd.DataFrame({"cluster": ukuran.index, "jumlah": ukuran.values})
            st.altair_chart(
                alt.Chart(df_uk).mark_bar().encode(
                    x=alt.X("cluster:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("jumlah:Q", title="Jumlah postingan"),
                    color=alt.Color("cluster:N", scale=skala_c, legend=None),
                    tooltip=["cluster", "jumlah"],
                ),
                width="stretch",
            )
        with c2:
            if "sentimen" in dc.columns and dc["sentimen"].notna().any():
                st.markdown("**Komposisi sentimen per cluster**")
                kom = dc.dropna(subset=["sentimen"]).groupby(["nama", "sentimen"]).size().reset_index(name="jumlah")
                st.altair_chart(
                    alt.Chart(kom).mark_bar().encode(
                        x=alt.X("nama:N", title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("jumlah:Q", stack="normalize", title="Proporsi", axis=alt.Axis(format="%")),
                        color=alt.Color("sentimen:N", scale=SKALA_WARNA, title="Sentimen"),
                        tooltip=["nama", "sentimen", "jumlah"],
                    ),
                    width="stretch",
                )
            else:
                st.markdown("**Komposisi sentimen per cluster**")
                st.caption("Tersedia setelah hasil pelabelan sentimen dibuat (lihat tab Sentimen).")

        # ---- kata dominan
        st.divider()
        st.subheader("Kata dominan per cluster")
        kolom = st.columns(len(ids))
        for kol, i in zip(kolom, ids):
            with kol:
                st.markdown(f"**Cluster {i}**")
                st.altair_chart(grafik_bobot_kata(hasil_cluster["kata"][i], warna_c[f"Cluster {i}"]),
                                width="stretch")

        # ---- word cloud
        st.divider()
        st.subheader("Word cloud")
        st.caption("Ukuran kata menunjukkan frekuensi kemunculan pada teks bersih (setelah stopword dibuang "
                   "dan stemming). Grafik batang di atas memakai bobot TF-IDF sehingga urutannya bisa sedikit berbeda.")
        peta_warna = {"Semua": "viridis", **{f"Cluster {i}": cm for i, cm in
                                            zip(ids, ["Blues", "Oranges", "Purples", "GnBu", "Reds"])}}
        tab_wc = st.tabs(list(peta_warna))
        for tab_i, nama_wc in zip(tab_wc, peta_warna):
            with tab_i:
                if nama_wc == "Semua":
                    teks_wc = " ".join(dc["clean_cluster"].astype(str))
                else:
                    teks_wc = " ".join(dc.loc[dc["nama"] == nama_wc, "clean_cluster"].astype(str))
                gambar, top5_wc = buat_wordcloud(teks_wc, peta_warna[nama_wc])
                if gambar is None:
                    st.caption("Tidak ada teks.")
                else:
                    st.image(gambar, width="stretch")
                    st.caption("5 kata teratas: " + ", ".join(top5_wc))

        # ---- contoh postingan
        st.divider()
        st.subheader("Contoh postingan")
        pilih = st.radio("Pilih cluster", ids, format_func=lambda i: f"Cluster {i}", horizontal=True)
        sub = dc[dc["cluster"] == pilih].sort_values("jarak")
        st.caption(f"**{sub['tema_pendek'].iloc[0]}** · 3 postingan paling dekat dengan pusat cluster "
                   "(paling mewakili):")
        konfigurasi = {
            "text": st.column_config.TextColumn("Postingan", width="large"),
            "like_count": "Likes",
            "jarak": st.column_config.NumberColumn("Jarak ke pusat", format="%.3f"),
        }
        st.dataframe(sub[["text", "like_count", "jarak"]].head(3), hide_index=True, width="stretch",
                     column_config=konfigurasi)
        with st.expander(f"Semua {len(sub)} postingan di Cluster {pilih}"):
            st.dataframe(sub[["text", "like_count", "jarak"]], hide_index=True, width="stretch",
                         column_config=konfigurasi)

        # ---- PCA
        st.divider()
        st.subheader("Visualisasi cluster (PCA 2D)")
        titik = alt.Chart(dc.assign(cuplikan=dc["text"].str.slice(0, 140))).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X("pca1:Q", title="Komponen utama 1"),
            y=alt.Y("pca2:Q", title="Komponen utama 2"),
            color=alt.Color("nama:N", scale=skala_c, title="Cluster"),
            tooltip=["nama", "tema_pendek", "cuplikan", "like_count"],
        )
        pusat = alt.Chart(hasil_cluster["titik_pusat"]).mark_point(
            shape="cross", size=250, strokeWidth=3, color="black").encode(x="pca1:Q", y="pca2:Q",
                                                                          tooltip=["cluster"])
        st.altair_chart((titik + pusat).properties(height=420), width="stretch")
        st.caption(
            f"Dua komponen utama hanya menangkap {hasil_cluster['var_pca'] * 100:.1f}% variasi data, "
            "sehingga tumpang tindih pada gambar tidak sepenuhnya mencerminkan jarak sebenarnya. "
            "Tanda silang hitam adalah pusat cluster."
        )

        st.download_button(
            "Unduh hasil clustering (CSV)",
            dc.drop(columns=[c for c in ("jarak", "pca1", "pca2", "tema_pendek", "nama", "sentimen")
                             if c in dc.columns]).to_csv(index=False).encode("utf-8"),
            file_name="hasil_clustering.csv", mime="text/csv",
        )


# ==========================================================================
# TAB REGRESI
# ==========================================================================
with tab_regresi:
    st.subheader("Regresi linear berganda: memperkirakan jumlah likes")
    st.markdown(
        "Model memperkirakan **jumlah likes (Y)** dari **jumlah reply (X1)** dan "
        "**jumlah retweet (X2)**. Data dibagi 80% latih dan 20% uji (`random_state=42`)."
    )

    if model_reg is None:
        st.error(f"Model tidak dapat dimuat dari `{PATH_MODEL.relative_to(BASE)}`. "
                 "Pastikan file ada dan `scikit-learn` terpasang.")
    else:
        koef = np.ravel(model_reg.coef_)
        intersep = float(model_reg.intercept_)

        # ---- Persamaan model ------------------------------------------------
        st.markdown("**Persamaan model**")
        st.latex(rf"\hat{{Y}} = {intersep:.2f} + {koef[0]:.2f}\,X_1 + {koef[1]:.2f}\,X_2")
        c1, c2, c3 = st.columns(3)
        c1.metric("Intersep", f"{intersep:.2f}")
        c2.metric("Koefisien reply (X1)", f"{koef[0]:.2f}")
        c3.metric("Koefisien retweet (X2)", f"{koef[1]:.2f}")
        st.caption(
            "Koefisien menunjukkan perubahan rata-rata likes untuk tambahan satu reply atau satu retweet, "
            "dengan variabel lain tetap. Ini hubungan statistik, bukan sebab-akibat; reply dan retweet "
            "saling berkorelasi tinggi sehingga pengaruh masing-masing sulit dipisahkan."
        )

        # ---- Simulasi ---------------------------------------------------------
        st.divider()
        st.subheader("Simulasi prediksi")
        with st.form("form_simulasi"):
            f1, f2 = st.columns(2)
            reply = f1.number_input("Jumlah reply (X1)", min_value=0, value=5, step=1)
            retweet = f2.number_input("Jumlah retweet (X2)", min_value=0, value=10, step=1)
            kirim = st.form_submit_button("Prediksi likes")

        mentah = prediksi_manual(model_reg, [reply, retweet])
        hasil_pred = max(0, round(mentah))
        p1, p2 = st.columns([1, 2])
        p1.metric("Prediksi jumlah likes", f"{hasil_pred}")
        with p2:
            rincian = pd.DataFrame({
                "Komponen": ["Intersep", f"Reply: {reply} × {koef[0]:.2f}",
                             f"Retweet: {retweet} × {koef[1]:.2f}", "Total (mentah)"],
                "Nilai": [intersep, reply * koef[0], retweet * koef[1], mentah],
            })
            st.dataframe(rincian.style.format({"Nilai": "{:.2f}"}), hide_index=True, width="stretch")
        if mentah < 0:
            st.caption("Hasil mentah model negatif (intersep negatif dan masukan kecil), "
                       "sehingga ditampilkan 0 karena likes tidak mungkin negatif.")
        st.caption("Prediksi akan kurang andal untuk nilai di luar rentang data latih.")

        # ---- Evaluasi ---------------------------------------------------------
        st.divider()
        st.subheader("Evaluasi model")

        med_likes = df["like_count"].median()
        if ada_data_regresi:
            ev = evaluasi_regresi(df, model_reg)
            mae, r2, korelasi = ev["mae"], ev["r2"], ev["korelasi"]
            if not ev["sama"]:
                st.warning(
                    "Koefisien hasil pelatihan ulang pada dataset ini berbeda dari model tersimpan, "
                    "jadi dataset ini kemungkinan bukan data yang dipakai melatih model. "
                    "Metrik di bawah dihitung dari model tersimpan pada 20% data uji dataset ini."
                )
            st.caption(f"Data latih {ev['n_latih']} postingan, data uji {ev['n_uji']} postingan.")
        else:
            mae, r2, korelasi, ev = HASIL_NOTEBOOK["mae"], HASIL_NOTEBOOK["r2"], matriks_korelasi_notebook(), None
            st.info(
                "Dataset yang dimuat belum memiliki kolom `reply_count` dan `retweet_count`, "
                "sehingga angka di bawah adalah **hasil dari notebook**. Ganti `data/data_clean.csv` "
                "dengan versi 7 kolom (yang memuat kedua kolom itu) agar evaluasi, grafik, dan tabel "
                "dihitung langsung di dashboard."
            )

        m1, m2 = st.columns(2)
        m1.metric("MAE", f"{mae:.2f}", help="Rata-rata selisih absolut antara likes aktual dan prediksi.")
        m2.metric("R²", f"{r2:.4f}", help="Proporsi variasi likes yang dijelaskan model (maks. 1).")
        st.caption(
            f"R² {r2:.2f} berarti sekitar {r2 * 100:.0f}% variasi likes pada data uji dijelaskan model. "
            f"MAE {mae:.0f} likes jauh di atas median likes dataset ({med_likes:.0f}), karena sebaran likes "
            "sangat condong dan beberapa postingan viral membuat selisih besar."
        )

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Korelasi likes, reply, dan retweet**")
            st.altair_chart(grafik_korelasi(korelasi), width="stretch")
        with g2:
            st.markdown("**Likes aktual vs prediksi (data uji)**")
            if ev is None:
                st.caption("Grafik ini tersedia setelah dataset memuat `reply_count` dan `retweet_count`.")
            else:
                simetris = st.checkbox("Skala simetris-log (lebih jelas untuk nilai kecil)", value=False)
                skala = alt.Scale(type="symlog") if simetris else alt.Scale()
                hasil = ev["hasil"]
                batas = [min(hasil["aktual"].min(), hasil["prediksi"].min()),
                         max(hasil["aktual"].max(), hasil["prediksi"].max())]
                titik = alt.Chart(hasil.reset_index(drop=True)).mark_circle(size=60, opacity=0.6, color="#2E8B57").encode(
                    x=alt.X("aktual:Q", title="Likes aktual", scale=skala),
                    y=alt.Y("prediksi:Q", title="Prediksi likes", scale=skala),
                    tooltip=[alt.Tooltip("aktual:Q"), alt.Tooltip("prediksi:Q", format=".1f"),
                             "reply_count", "retweet_count"],
                )
                garis = alt.Chart(pd.DataFrame({"x": batas, "y": batas})).mark_line(
                    color="#D1495B", strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
                st.altair_chart(titik + garis, width="stretch")
                st.caption("Garis putus-putus merah: prediksi sempurna (aktual = prediksi).")

        if ev is not None:
            with st.expander("Lihat data uji beserta selisih prediksi"):
                tabel = ev["hasil"].sort_values("selisih", key=np.abs, ascending=False)
                st.dataframe(
                    tabel[["text", "reply_count", "retweet_count", "aktual", "prediksi", "selisih"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "text": st.column_config.TextColumn("Postingan", width="large"),
                        "reply_count": "Reply", "retweet_count": "Retweet",
                        "aktual": "Likes aktual",
                        "prediksi": st.column_config.NumberColumn("Prediksi", format="%.1f"),
                        "selisih": st.column_config.NumberColumn("Selisih (prediksi − aktual)", format="%.1f"),
                    },
                )
