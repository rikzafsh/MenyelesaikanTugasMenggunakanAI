# Dashboard Analisis Sentimen dan Tema Pembahasan Penggunaan AI untuk Tugas Kuliah

Aplikasi Streamlit untuk menampilkan hasil analisis postingan X berbahasa Indonesia tentang
penggunaan AI untuk membuat dan menyelesaikan tugas kuliah. Aplikasi membaca dataset CSV yang
sudah disiapkan (tidak ada pengambilan data real-time).

**Status saat ini:** tab **Data** dan **Sentimen**. Tab Clustering dan Regresi belum ditambahkan.

## Struktur folder

```
.
├── app.py                    # aplikasi Streamlit
├── label_sentiment.py        # pelabelan sentimen (jalankan sekali)
├── requirements.txt          # dependensi aplikasi (ringan)
├── requirements-label.txt    # dependensi tambahan untuk label_sentiment.py
├── README.md
└── data/
    ├── data_clean.csv            # dataset bersih (192 postingan)
    ├── data_sentiment.csv        # dibuat oleh label_sentiment.py
    └── evaluasi_sentiment.csv    # dibuat oleh label_sentiment.py
```

## Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Buat hasil sentimen (sekali saja; butuh internet untuk mengunduh model)
pip install -r requirements-label.txt
python label_sentiment.py

# 2) Jalankan dashboard
streamlit run app.py
```

Tanpa langkah 1, tab Data tetap berfungsi dan tab Sentimen menampilkan petunjuk.

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
| `data_clean.csv` | `text`, `clean_text`, `like_count`, `hashtag_count`, `created_at` |
| `data_sentiment.csv` | kolom di atas + `sentiment` (`positive`/`neutral`/`negative`), `sentiment_score` (opsional) |
| `evaluasi_sentiment.csv` | `text`, `sentiment` (prediksi sistem), `label_manual` |

## Metode singkat

- **Persiapan data:** regex untuk menghapus tautan, *mention*, angka, tanda baca, dan emoji;
  kata negasi dipertahankan; postingan promosi joki dan di luar topik dibuang.
- **Sentimen:** model pre-trained `mdhugol/indonesia-bert-sentiment-classification` pada `clean_text`,
  divalidasi manual pada 20 sampel acak (`random_state=42`).
