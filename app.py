"""
Dashboard Analisis Sentimen dan Tema Pembahasan Penggunaan AI untuk Membuat
dan Menyelesaikan Tugas pada Postingan X Berbahasa Indonesia.

Versi saat ini: tab Data dan Sentimen.
(Tab Clustering dan Regresi akan ditambahkan kemudian.)

Jalankan:  streamlit run app.py
"""
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
BASE = Path(__file__).parent
PATH_CLEAN = BASE / "data" / "data_clean.csv"
PATH_SENTIMENT = BASE / "data" / "data_sentiment.csv"
PATH_EVAL = BASE / "data" / "evaluasi_sentiment.csv"

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
    """Baca CSV; coba pemisah koma, lalu '|' (format simpanan notebook)."""
    df = pd.read_csv(path)
    if "clean_text" not in df.columns:
        df = pd.read_csv(path, sep="|")
    return df


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
    df = _baca_csv_bebas(p)
    if df is None or not {"sentiment", "label_manual"}.issubset(df.columns):
        return None
    return df


def _baca_csv_bebas(path: Path) -> pd.DataFrame | None:
    for sep in (",", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


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

tab_data, tab_sentimen = st.tabs(["Data", "Sentimen"])

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
