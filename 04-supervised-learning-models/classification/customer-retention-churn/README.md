# Klasifikasi Customer Churn

Eksperimen pemodelan pembelajaran terarah (supervised learning) untuk memprediksi potensi churn pelanggan e-commerce menggunakan Random Forest Classifier.

## Variabel yang Digunakan
- `hari_terakhir_aktif`: Jumlah hari sejak transaksi terakhir pelanggan (recency).
- `frekuensi_belanja`: Jumlah transaksi belanja dalam 3 bulan terakhir.
- `usia_akun_bulan`: Lama akun pelanggan terdaftar (bulan).
- `tiket_komplain_cs`: Frekuensi keluhan yang diajukan ke layanan pelanggan (CS).
- `status_churn` (Target): 0 = Pelanggan Aktif, 1 = Churn (Berhenti Belanja).

## Berkas dan Implementasi
- `prediksi_retensi_churn.py`: Skrip mandiri pemodelan Random Forest, evaluasi metrik (Precision, Recall, ROC-AUC), dan kalkulasi feature importance.
- `prediksi_retensi_churn.ipynb`: Notebook Jupyter interaktif pengujian pemodelan churn.
