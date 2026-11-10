# Eksperimen Data Mining Terapan

Repositori ini memuat implementasi dan pengujian alur kerja data mining berbasis Python, mencakup tahapan pembersihan data mentah, eksplorasi karakteristik sebaran data, rekayasa fitur, hingga pemodelan prediktif dan klasterisasi.

---

## Ringkasan Modul

- **01. Data Cleaning** (`01-data-cleaning/`)
  Pembersihan data transaksi e-commerce: penanganan duplikasi baris, outlier capping dengan metode Interquartile Range (IQR), serta imputasi nilai kosong menggunakan KNN Imputer dan modus.

- **02. Data Exploration** (`02-data-exploration/`)
  Analisis karakteristik statistik dan visualisasi bentuk sebaran frekuensi total pengeluaran tahunan pelanggan retail menggunakan histogram dan kurva KDE.

- **03. Feature Transformation** (`03-feature-transformation/`)
  Transformasi fitur data pesanan e-commerce: pembuatan fitur rasio belanja per item, transformasi logaritma np.log1p(), serta penggabungan MinMaxScaler dan One-Hot Encoding melalui ColumnTransformer.

- **04. Supervised Learning Models** (`04-supervised-learning-models/`)
  - Regresi: Estimasi harga pasar laptop berdasarkan spesifikasi perangkat keras (RAM, SSD, prosesor) menggunakan Linear Regression.
  - Klasifikasi: Prediksi potensi customer churn pelanggan menggunakan Random Forest Classifier lengkap dengan evaluasi Precision, Recall, ROC-AUC, dan feature importance.

- **05. Unsupervised Clustering** (`05-unsupervised-clustering/`)
  Segmentasi kelompok perilaku pembeli digital berdasarkan intensitas kunjungan aplikasi dan nilai rata-rata transaksi menggunakan algoritma K-Means dan evaluasi Silhouette Score.

---

## Cara Menjalankan

Pastikan dependensi pustaka sudah terpasang di lingkungan Python:
```bash
pip install -r requirements.txt
```

Setiap modul dapat dijalankan langsung melalui skrip mandiri atau dieksplorasi secara interaktif menggunakan Jupyter Notebook:

```bash
# Menjalankan modul 01 (pembersihan data)
python 01-data-cleaning/cleaning_transaksi_ecommerce.py

# Menjalankan modul 02 (eksplorasi data)
python 02-data-exploration/retail-spending-analysis/analisis_belanja_retail.py

# Menjalankan modul 03 (transformasi fitur)
python 03-feature-transformation/transformasi_fitur_order.py

# Menjalankan modul 04 (regresi laptop)
python 04-supervised-learning-models/regression/laptop-price-estimation/estimasi_harga_laptop.py

# Menjalankan modul 04 (klasifikasi churn)
python 04-supervised-learning-models/classification/customer-retention-churn/prediksi_retensi_churn.py

# Menjalankan modul 05 (clustering k-means)
python 05-unsupervised-clustering/shopper-behavior-clustering/segmentasi_perilaku_belanja.py
```

Untuk membuka lembar kerja interaktif:
```bash
jupyter lab
```

---

## Kebutuhan Sistem
Seluruh pustaka pendukung (pandas, numpy, scipy, scikit-learn, matplotlib, seaborn) tercatat pada berkas `requirements.txt`.
