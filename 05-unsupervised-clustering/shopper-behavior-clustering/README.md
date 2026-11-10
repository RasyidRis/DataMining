# Segmentasi Perilaku Pembeli E-Commerce (K-Means)

Eksperimen pembelajaran tanpa label (unsupervised learning) untuk mengelompokkan pola perilaku pembeli digital berdasarkan intensitas kunjungan dan nilai belanja menggunakan algoritma K-Means Clustering.

## Variabel yang Digunakan
- `kunjungan_bulanan`: Frekuensi membuka atau mengunjungi aplikasi per bulan.
- `nilai_belanja_ribu`: Rata-rata nilai transaksi keranjang belanja dalam ribuan IDR.
- `kluster` (Hasil Segmentasi): Label kelompok pelanggan hasil klasterisasi K-Means.

## Berkas dan Implementasi
- `segmentasi_perilaku_belanja.py`: Skrip mandiri pemodelan K-Means, evaluasi Silhouette Score, dan visualisasi scatter plot titik centroid.
- `segmentasi_perilaku_belanja.ipynb`: Notebook Jupyter interaktif pengujian pemodelan segmentasi.
