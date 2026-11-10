# 01. Data Cleaning Pipeline

Bagian ini membahas tahapan fundamental data preprocessing: pembersihan dan penyiapan data mentah (data cleaning & preparation) dengan studi kasus data transaksi dan aktivitas pelanggan e-commerce.

---

## Latar Belakang Masalah
Dalam data transaksi dunia nyata (real-world datasets), data mentah jarang sekali langsung siap digunakan untuk pemodelan machine learning. Sering kali ditemukan:
- Duplikasi Transaksi: Pengiriman berulang (network retry) atau kesalahan pencatatan sistem.
- Inkonsistensi Format String: Perbedaan kapitalisasi (misal bronze, Silver, GOLD), spasi berlebih (leading/trailing whitespace), dan nilai sentinel (seperti None, ?, Unknown).
- Pelanggaran Batasan Domain: Usia bernilai negatif atau di atas batas biologis logis, total belanja bernilai minus, dan rating di luar rentang valid.
- Pencilan Ekstrem (Outliers): Transaksi dengan nilai belanja yang sangat besar dan mendistorsi fungsi kerugian (loss function) algoritma.
- Nilai Kosong (Missing Data): Pelanggan yang tidak melengkapi profil atau kegagalan pencatatan transaksi.

---

## Alur Pipeline Pembersihan

Pipeline pada modul ini dirancang secara berurutan dan terstruktur:

```text
[Data Mentah (Raw Data)]
         │
         ▼
 1. Deduplikasi Baris (drop_duplicates berdasarkan order_id)
         │
         ▼
 2. Standardisasi Teks (strip whitespace, title-case, validasi kategori resmi)
         │
         ▼
 3. Validasi Batasan Nilai Domain (set nilai anomali/out-of-range ke NaN)
         │
         ▼
 4. Outlier Capping (Metode Interquartile Range / IQR Winsorization)
         │
         ▼
 5. Imputasi Nilai Kosong (Modus untuk Kategori, Median / KNN untuk Numerik)
         │
         ▼
[Data Bersih & Siap Digunakan (Cleaned Dataset)]
```

---

## Kamus Data (Data Dictionary)

| Kolom | Tipe Data | Deskripsi | Aturan Validasi / Cleaning |
|---|---|---|---|
| `order_id` | String | Identitas unik transaksi pesanan | Harus unik, duplikasi dihapus |
| `customer_age` | Integer | Usia pelanggan | Rentang logis 17–85 tahun |
| `membership_tier` | Kategori | Tingkatan loyalitas pelanggan | Kategori baku: Bronze, Silver, Gold, Platinum |
| `total_spend_k` | Float | Nilai total transaksi (ribuan IDR) | > 0, outlier di-cap dengan batas atas IQR |
| `items_count` | Integer | Jumlah barang yang dibeli | Minimal 1 barang |
| `delivery_distance_km` | Float | Jarak lokasi pengiriman (km) | > 0 km, missing diimputasi |
| `customer_rating` | Float | Skor penilaian kepuasan | Skala valid 1.0 – 5.0 |

---

## Berkas dan Implementasi

- `cleaning_transaksi_ecommerce.py`: Skrip Python mandiri modular berisi generator data mentah, fungsi pipeline pembersihan, dan laporan evaluasi kualitas data.
- `cleaning_transaksi_ecommerce.ipynb`: Notebook Jupyter interaktif langkah demi langkah untuk visualisasi dan pengujian.

---

## Cara Menjalankan

### 1. Menjalankan Skrip Python
```bash
python 01-data-cleaning/cleaning_transaksi_ecommerce.py
```

### 2. Menjalankan Jupyter Notebook
```bash
jupyter lab 01-data-cleaning/cleaning_transaksi_ecommerce.ipynb
```

---

## Ringkasan Hasil Eksperimen

- Deduplikasi: Mereduksi 4 baris duplikat redundan.
- Missing Values: 100% nilai kosong teratasi tanpa mendistorsi pola distribusi data awal.
- Penanganan Outlier: Nilai belanja ekstrem (hingga >15.000k) distabilkan menggunakan batas atas IQR (Q3 + 1.5 * IQR) menjadi ~1.158k sehingga tidak mendominasi gradien saat pelatihan model.
