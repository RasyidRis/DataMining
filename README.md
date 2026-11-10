# Practical Data Mining & Machine Learning Repository

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/notebooks-JupyterLab-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)

Repositori terstruktur untuk pembelajaran dan eksperimen Data Mining, Machine Learning, dan Analitik Data berbasis Python.

---

## Struktur Repositori

```text
.
├── 01-data-cleaning/
│   ├── cleaning_transaksi_ecommerce.py       # Pipeline pembersihan data mandiri
│   ├── cleaning_transaksi_ecommerce.ipynb    # Notebook eksplorasi interaktif
│   └── README.md                             # Dokumentasi teknis modul 01
├── 02-data-exploration/
│   ├── retail-spending-analysis/
│   │   ├── analisis_belanja_retail.py        # Skrip statistik deskriptif & visualisasi histogram
│   │   ├── analisis_belanja_retail.ipynb     # Notebook interaktif analisis sebaran data
│   │   └── README.md                         # Dokumentasi submodul belanja retail
│   └── README.md                             # Dokumentasi modul 02
├── 03-feature-transformation/
│   ├── transformasi_fitur_order.py           # Skrip rekayasa fitur & ColumnTransformer
│   ├── transformasi_fitur_order.ipynb        # Notebook interaktif transformasi fitur
│   └── README.md                             # Dokumentasi teknis modul 03
├── 04-supervised-learning-models/
│   ├── regression/
│   │   └── laptop-price-estimation/          # Estimasi harga laptop (Linear Regression)
│   ├── classification/
│   │   └── customer-retention-churn/         # Klasifikasi customer churn (Random Forest)
│   └── README.md                             # Dokumentasi teknis modul 04
├── 05-unsupervised-clustering/
│   ├── shopper-behavior-clustering/
│   │   ├── segmentasi_perilaku_belanja.py    # Skrip segmentasi pembeli (K-Means)
│   │   ├── segmentasi_perilaku_belanja.ipynb # Notebook interaktif klasterisasi
│   │   └── README.md                         # Dokumentasi teknis submodul clustering
│   └── README.md                             # Dokumentasi teknis modul 05
├── requirements.txt                          # Daftar dependensi pustaka
├── .gitignore                                # Konfigurasi pengecualian file git
└── README.md                                 # Dokumentasi utama repositori
```

---

## Daftar Modul

| No | Modul | Topik Inti | Algoritma / Metode | Studi Kasus |
|---|---|---|---|---|
| 01 | [Data Cleaning](./01-data-cleaning/) | Handling Missing Values, Outlier Treatment, Normalisasi Kategori | IQR Capping, Mode Imputation, KNN Imputer | Transaksi Pelanggan Retail / E-Commerce |
| 02 | [Data Exploration](./02-data-exploration/) | Analisis Statistik Deskriptif, Karakteristik Sebaran Data | Histogram Distribusi, Kurva KDE Seaborn | Analisis Pengeluaran Belanja Pelanggan |
| 03 | [Feature Transformation](./03-feature-transformation/) | Categorical Encoding, Numerical Scaling, Feature Creation | One-Hot Encoding, MinMaxScaler, Log Transformation | Rekayasa Fitur Pesanan & Pengiriman E-Commerce |
| 04 | [Supervised Learning Models](./04-supervised-learning-models/) | Pemodelan Prediksi Regresi & Klasifikasi Terarah | Linear Regression, Random Forest Classifier | Estimasi Harga Laptop & Prediksi Customer Churn |
| 05 | [Unsupervised Clustering](./05-unsupervised-clustering/) | Segmentasi Kelompok Tanpa Label, Analisis Profil Klaster | K-Means Clustering, Silhouette Score Analysis | Segmentasi Perilaku Pembeli E-Commerce |

---

## Panduan Menjalankan Proyek

### 1. Kloning Repositori
```bash
git clone <URL_REPOSITORI_ANDA>
cd <NAMA_FOLDER_REPOSITORI>
```

### 2. Konfigurasi Virtual Environment
```bash
# Membuat environment virtual
python -m venv venv

# Mengaktifkan di Windows:
venv\Scripts\activate

# Mengaktifkan di macOS / Linux:
source venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Kode Eksperimen
- **Modul 01 (Data Cleaning):**
  ```bash
  python 01-data-cleaning/cleaning_transaksi_ecommerce.py
  ```
- **Modul 02 (Data Exploration):**
  ```bash
  python 02-data-exploration/retail-spending-analysis/analisis_belanja_retail.py
  ```
- **Modul 03 (Feature Transformation):**
  ```bash
  python 03-feature-transformation/transformasi_fitur_order.py
  ```
- **Modul 04 - Regresi:**
  ```bash
  python 04-supervised-learning-models/regression/laptop-price-estimation/estimasi_harga_laptop.py
  ```
- **Modul 04 - Klasifikasi:**
  ```bash
  python 04-supervised-learning-models/classification/customer-retention-churn/prediksi_retensi_churn.py
  ```
- **Modul 05 (Unsupervised Clustering):**
  ```bash
  python 05-unsupervised-clustering/shopper-behavior-clustering/segmentasi_perilaku_belanja.py
  ```

---

## Pustaka dan Perangkat Lunak
- Manipulasi dan Analisis Data: `pandas`, `numpy`, `scipy`
- Machine Learning & Preprocessing: `scikit-learn`
- Visualisasi Data: `matplotlib`, `seaborn`
- Lingkungan Interaktif: `jupyter`, `jupyterlab`

---

## Lisensi
Repositori ini ditujukan untuk sarana pembelajaran, riset, dan eksplorasi teknis mandiri.
