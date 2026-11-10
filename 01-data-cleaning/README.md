# 01. Data Cleaning

Pembersihan dan penyiapan data transaksi e-commerce sebelum masuk ke tahap analisis dan pemodelan.

## Topik Utama
- Deduplikasi: Menghapus baris transaksi kembar berdasarkan id_transaksi.
- Standardisasi Teks: Merapikan format kapitalisasi kategori membership.
- Validasi Batasan: Mengoreksi anomali nilai usia dan rating di luar rentang wajar.
- Outlier Capping: Membatasi nilai belanja ekstrem menggunakan metode Interquartile Range (IQR).
- Imputasi Nilai Kosong: Mengisi nilai null dengan modus untuk kategori dan KNN Imputer untuk angka.

## Berkas
- `cleaning_transaksi_ecommerce.py`: Skrip pipeline pembersihan data mandiri.
- `cleaning_transaksi_ecommerce.ipynb`: Notebook pengujian interaktif langkah demi langkah.
