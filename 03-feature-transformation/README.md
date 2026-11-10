# 03. Feature Transformation

Bagian ini membahas teknik transformasi dan rekayasa fitur data mining agar model Machine Learning dapat mengekstrak pola secara optimal dengan studi kasus data transaksi dan pengiriman e-commerce.

---

## Topik Utama

1. Categorical Encoding:
   - One-Hot Encoding untuk fitur nominal tanpa tingkatan (wilayah pengiriman dan tipe layanan kirim).
   - Penggunaan parameter `drop="first"` untuk mencegah multikolinearitas (dummy variable trap).

2. Feature Scaling:
   - MinMaxScaler untuk menstandardisasi rentang fitur numerik ke skala seragam [0, 1].

3. Feature Creation & Transformation:
   - Rasio Interaksi: Pembuatan fitur `belanja_per_item` (rata-rata harga per unit barang).
   - Log Transformation: Menerapkan fungsi logaritma `np.log1p()` untuk meredam skewness nilai moneter.

4. ColumnTransformer Pipeline:
   - Penggabungan transformer numerik dan kategorikal ke dalam satu pipeline preprocessing terpadu Scikit-Learn.

---

## Berkas dan Implementasi

- `transformasi_fitur_order.py`: Skrip mandiri pemrosesan rekayasa fitur menggunakan ColumnTransformer.
- `transformasi_fitur_order.ipynb`: Notebook interaktif demonstrasi transformasi fitur langkah demi langkah.

---

## Cara Menjalankan

### 1. Menjalankan Skrip Python
```bash
python 03-feature-transformation/transformasi_fitur_order.py
```

### 2. Menjalankan Jupyter Notebook
```bash
jupyter lab 03-feature-transformation/transformasi_fitur_order.ipynb
```
