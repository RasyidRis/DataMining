# Segmentasi Perilaku Pembeli E-Commerce

Pengelompokan profil pembeli digital berdasarkan intensitas kunjungan dan nilai transaksi belanja menggunakan algoritma K-Means Clustering.

## Variabel
- `kunjungan_bulanan`: Frekuensi membuka atau mengunjungi aplikasi per bulan.
- `nilai_belanja_ribu`: Rata-rata nilai transaksi belanja (ribu IDR).
- `kluster` (Hasil): Label segmen kelompok pelanggan (0, 1, 2).

## Berkas
- `segmentasi_perilaku_belanja.py`: Skrip pemodelan K-Means, evaluasi Silhouette Score, dan visualisasi scatter plot titik centroid.
- `segmentasi_perilaku_belanja.ipynb`: Notebook pengujian pemodelan segmentasi.
