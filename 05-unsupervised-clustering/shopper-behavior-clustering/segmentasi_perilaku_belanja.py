"""
Unsupervised Learning: Segmentasi Perilaku Pembeli E-Commerce (K-Means)
Mengelompokkan pelanggan digital berdasarkan frekuensi kunjungan dan nilai transaksi rata-rata.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_shopper_cluster_dataset() -> pd.DataFrame:
    """Membangkitkan data sintetis perilaku belanja untuk segmentasi klaster."""
    np.random.seed(42)

    # 3 Kelompok Profil Pembeli:
    # 1. Window Shoppers: Sering buka aplikasi, belanja kecil
    c1 = np.random.multivariate_normal([35, 120], [[15, 0], [0, 600]], 60)
    # 2. Regular Buyers: Kunjungan sedang, belanja sedang
    c2 = np.random.multivariate_normal([18, 380], [[12, 0], [0, 1500]], 70)
    # 3. High-Value Spenders: Kunjungan terarah, nilai belanja tinggi
    c3 = np.random.multivariate_normal([8, 850], [[8, 0], [0, 4000]], 50)

    data_gabungan = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data_gabungan, columns=["kunjungan_bulanan", "nilai_belanja_ribu"])
    df["kunjungan_bulanan"] = df["kunjungan_bulanan"].clip(1, 60).round().astype(int)
    df["nilai_belanja_ribu"] = df["nilai_belanja_ribu"].clip(30, 1500).round(1)
    return df


def run_kmeans_clustering(df: pd.DataFrame):
    """Melatih model K-Means clustering, menghitung metrik evaluasi, dan visualisasi."""
    X = df[["kunjungan_bulanan", "nilai_belanja_ribu"]]

    # Model K-Means dengan 3 klaster
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_hasil = df.copy()
    df_hasil["kluster"] = kmeans.fit_predict(X)

    skor_sil = silhouette_score(X, df_hasil["kluster"])

    print("=== EVALUASI SEGMENTASI PELANGGAN (K-MEANS) ===")
    print(f"Jumlah sampel data : {len(df_hasil)} pelanggan")
    print(f"Jumlah klaster (k) : 3 klaster")
    print(f"Skor Silhouette    : {skor_sil:.4f}\n")

    print("Profil Rata-rata per Klaster:")
    profil = df_hasil.groupby("kluster")[["kunjungan_bulanan", "nilai_belanja_ribu"]].mean().round(1)
    profil["jumlah_anggota"] = df_hasil["kluster"].value_counts().sort_index()
    print(profil.to_string())

    # Visualisasi Scatter Plot Hasil Klasterisasi
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df_hasil["kunjungan_bulanan"], 
        df_hasil["nilai_belanja_ribu"], 
        c=df_hasil["kluster"], 
        cmap="plasma", 
        edgecolors="k", 
        alpha=0.8
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0], 
        kmeans.cluster_centers_[:, 1], 
        s=200, 
        c="red", 
        marker="X", 
        label="Centroid Klaster"
    )
    plt.title("Segmentasi Perilaku Pembeli E-Commerce (K-Means)", pad=12)
    plt.xlabel("Frekuensi Kunjungan Aplikasi per Bulan")
    plt.ylabel("Rata-rata Nilai Belanja (Ribu IDR)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_shopper = build_shopper_cluster_dataset()
    run_kmeans_clustering(data_shopper)
