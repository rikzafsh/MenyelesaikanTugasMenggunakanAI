# Dashboard Analisis Sentimen dan Tema Pembahasan Penggunaan AI untuk Tugas Kuliah

Aplikasi Streamlit untuk menampilkan hasil analisis postingan X berbahasa Indonesia tentang
penggunaan AI untuk membuat dan menyelesaikan tugas kuliah. Aplikasi membaca dataset CSV yang
sudah disiapkan (tidak ada pengambilan data real-time).

**Status saat ini:** empat tab: **Data**, **Sentimen**, **Clustering**, dan **Regresi**.

## Struktur folder

```
.
├── app.py                    # aplikasi Streamlit
├── label_sentiment.py        # pelabelan sentimen (jalankan sekali)
├── cluster_teks.py           # clustering teks TF-IDF + K-Means (jalankan sekali)
├── model/
│   └── model_regresi_ai.joblib   # model regresi linear (reply, retweet -> likes)
├── requirements.txt          # dependensi aplikasi (ringan)
├── requirements-label.txt    # dependensi tambahan untuk label_sentiment.py
├── README.md
└── data/
    ├── data_clean.csv            # dataset bersih (192 postingan); koma atau '|' keduanya dibaca
    ├── data_sentiment.csv        # dibuat oleh label_sentiment.py
    └── evaluasi_sentiment.csv    # dibuat oleh label_sentiment.py
```

## Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Buat hasil sentimen dan clustering (sekali saja; butuh internet untuk mengunduh model/stopword)
pip install -r requirements-label.txt
python label_sentiment.py      # -> data/data_sentiment.csv, data/evaluasi_sentiment.csv
python cluster_teks.py         # -> data/hasil_clustering.csv

# 2) Jalankan dashboard
streamlit run app.py
```

Tanpa langkah 1, tab Data dan Regresi tetap berfungsi, sedangkan tab Sentimen dan Clustering menampilkan petunjuk.

### Alternatif: ekspor langsung dari notebook

Di akhir bagian sentimen notebook, tambahkan:

```python
df_label.to_csv('data_sentiment.csv', index=False)
df_tampil[['text', 'sentiment', 'label_manual']].to_csv('evaluasi_sentiment.csv', index=False)
```

lalu letakkan kedua file di folder `data/`.

## Format berkas

| Berkas | Kolom wajib |
|---|---|
| `data_clean.csv` | `text`, `clean_text`, `like_count`, `hashtag_count`, `created_at`; untuk evaluasi regresi tambahkan `reply_count`, `retweet_count` |
| `data_sentiment.csv` | kolom di atas + `sentiment` (`positive`/`neutral`/`negative`), `sentiment_score` (opsional) |
| `evaluasi_sentiment.csv` | `text`, `sentiment` (prediksi sistem), `label_manual` |
| `hasil_clustering.csv` | `text`, `clean_cluster`, `cluster`; opsional `tema`, `like_count` |

## Tab Clustering

- Pengelompokan tema memakai pembersihan + stemming Sastrawi, **TF-IDF** (`max_df=0.8`, `min_df=2`) dan
  **K-Means 3 cluster** (`random_state=42`), sama dengan notebook.
- Dashboard menampilkan ukuran cluster, tema dan kata dominan, contoh postingan paling dekat ke pusat cluster,
  skor silhouette, proyeksi PCA 2D, **word cloud** (keseluruhan dan per cluster), dan komposisi sentimen per cluster
  (bila hasil sentimen tersedia).
- Yang dibaca dashboard hanya kolom `text`, `clean_cluster`, `cluster` (opsional `tema`, `like_count`), jadi
  `hasil_clustering.csv` dari notebook dapat dipakai langsung. Silhouette, kata dominan, dan PCA dihitung ulang
  di dashboard dari kolom tersebut.
- Skor silhouette pada data ini sangat rendah (0,013): cluster saling tumpang tindih dan perlu dibaca
  sebagai kecenderungan tema, bukan kelompok yang terpisah tegas.

## Tab Regresi

- Model: regresi linear berganda, **X1 = `reply_count`, X2 = `retweet_count`, Y = `like_count`**
  (sesuai notebook), data latih/uji 80:20 dengan `random_state=42`.
- **Simulasi:** isi jumlah reply dan retweet, dashboard menampilkan prediksi likes memakai
  `model/model_regresi_ai.joblib` (tidak butuh dataset).
- **Evaluasi (MAE, R², heatmap korelasi, grafik aktual vs prediksi, tabel data uji):** dihitung langsung
  bila `data/data_clean.csv` memuat kolom `reply_count` dan `retweet_count` (versi 7 kolom dari notebook).
  Bila belum, dashboard menampilkan hasil dari notebook (MAE 81,19; R² 0,5916) dengan keterangan.
- Model disimpan dengan scikit-learn 1.6.1; versi yang lebih baru tetap dapat memuatnya
  (peringatan versi disembunyikan). Jika gagal dimuat, samakan dengan `pip install scikit-learn==1.6.1`.

## Metode singkat

- **Persiapan data:** regex untuk menghapus tautan, *mention*, angka, tanda baca, dan emoji;
  kata negasi dipertahankan; postingan promosi joki dan di luar topik dibuang.
- **Sentimen:** model pre-trained `mdhugol/indonesia-bert-sentiment-classification` pada `clean_text`,
  divalidasi manual pada 20 sampel acak (`random_state=42`).
- **Clustering:** TF-IDF + K-Means (3 cluster), dievaluasi dengan skor silhouette dan ditinjau lewat kata dominan,
  contoh postingan, serta proyeksi PCA.
- **Regresi linear berganda:** `LinearRegression` (scikit-learn), dievaluasi dengan MAE dan R² pada 20% data uji.
