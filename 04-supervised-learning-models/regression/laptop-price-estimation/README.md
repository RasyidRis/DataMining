# Estimasi Harga Laptop (Regresi)

Eksperimen pemodelan pembelajaran terarah (supervised learning) untuk memprediksi harga laptop berdasarkan fitur spesifikasi perangkat keras menggunakan Linear Regression.

## Variabel yang Digunakan
- `ram_gb`: Kapasitas memori RAM dalam Gigabyte (4, 8, 16, 32 GB).
- `storage_ssd_gb`: Kapasitas penyimpanan SSD (256, 512, 1024 GB).
- `skor_prosesor`: Nilai skor benchmark performa prosesor (CPU score).
- `harga_juta` (Target): Estimasi harga pasar laptop dalam satuan Juta IDR.

## Berkas dan Implementasi
- `estimasi_harga_laptop.py`: Skrip mandiri pelatihan model dan kalkulasi metrik evaluasi (R2, MAE, RMSE).
- `estimasi_harga_laptop.ipynb`: Notebook Jupyter interaktif pengujian pemodelan regresi.
