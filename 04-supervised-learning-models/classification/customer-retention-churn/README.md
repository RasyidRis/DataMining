# Klasifikasi Customer Churn

Prediksi potensi churn pelanggan e-commerce berdasarkan data perilaku transaksi menggunakan Random Forest Classifier.

## Variabel
- `hari_terakhir_aktif`: Jumlah hari sejak transaksi terakhir (recency).
- `frekuensi_belanja`: Jumlah transaksi belanja dalam 3 bulan terakhir.
- `usia_akun_bulan`: Lama akun pelanggan terdaftar (bulan).
- `tiket_komplain_cs`: Jumlah komplain ke layanan pelanggan.
- `status_churn` (Target): 0 = Aktif, 1 = Churn.

## Berkas
- `prediksi_retensi_churn.py`: Skrip pelatihan Random Forest, evaluasi metrik (Precision, Recall, ROC-AUC), dan analisis feature importance.
- `prediksi_retensi_churn.ipynb`: Notebook pengujian pemodelan klasifikasi.
