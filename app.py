"""
Dashboard Analisis Sentimen dan Tema Pembahasan Penggunaan AI untuk Membuat
dan Menyelesaikan Tugas pada Postingan X Berbahasa Indonesia.

Versi saat ini: tab Data, Sentimen, dan Regresi.
(Tab Clustering akan ditambahkan kemudian.)

Jalankan:  streamlit run app.py
"""
import warnings
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
BASE = Path(__file__).parent
PATH_CLEAN = BASE / "data" / "data_clean.csv"
PATH_SENTIMENT = BASE / "data" / "data_sentiment.csv"
PATH_EVAL = BASE / "data" / "evaluasi_sentiment.csv"
PATH_MODEL = BASE / "model" / "model_regresi_ai.joblib"

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
    st.markdown(f"- Model regresi: {'✅' if model_reg is not None else '❌ belum ada'}")
    st.markdown(f"- Data reply/retweet: {'✅' if ada_data_regresi else '❌ belum ada'}")

tab_data, tab_sentimen, tab_regresi = st.tabs(["Data", "Sentimen", "Regresi"])

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
