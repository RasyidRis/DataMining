# 03. Feature Transformation

Transformasi dan rekayasa fitur data pesanan e-commerce agar siap digunakan untuk pemodelan Machine Learning.

## Topik Utama
- Feature Creation: Membuat rasio rata-rata belanja per item barang (`belanja_per_item`).
- Log Transformation: Menerapkan fungsi logaritma `np.log1p()` untuk meredam kemiringan (skewness) nilai uang belanja.
- Feature Scaling & Encoding: Menggabungkan `MinMaxScaler` dan `OneHotEncoder` menggunakan `ColumnTransformer`.

## Berkas
- `transformasi_fitur_order.py`: Skrip pipeline rekayasa fitur modular.
- `transformasi_fitur_order.ipynb`: Notebook eksperimen transformasi fitur.
