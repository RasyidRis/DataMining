# Estimasi Harga Laptop (Regresi)

Prediksi harga laptop berdasarkan spesifikasi perangkat keras (RAM, kapasitas SSD, dan skor prosesor) menggunakan algoritma Linear Regression.

## Variabel
- `ram_gb`: Kapasitas RAM (GB).
- `storage_ssd_gb`: Kapasitas penyimpanan SSD (GB).
- `skor_prosesor`: Skor benchmark performa prosesor.
- `harga_juta` (Target): Estimasi harga laptop dalam juta IDR.

## Berkas
- `estimasi_harga_laptop.py`: Skrip pelatihan model dan kalkulasi metrik evaluasi (R2, MAE, RMSE).
- `estimasi_harga_laptop.ipynb`: Notebook pengujian pemodelan regresi.
