"""
Modul: 02-data-exploration
Topik: Analisis Eksplorasi Data (EDA) & Distribusi Variabel
Studi Kasus: Pola Pengeluaran Pelanggan Retail (Customer Spending Analytics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_customer_spending_dataset() -> pd.DataFrame:
    """Membangkitkan data sintetis aktivitas belanja dan profil pelanggan retail."""
    np.random.seed(42)
    n = 200

    durasi_member_bulan = np.random.randint(1, 48, n)
    pendapatan_juta = np.random.normal(12, 4.5, n).clip(3.5, 35.0).round(1)
    frekuensi_transaksi = np.random.poisson(lam=6.0, size=n).clip(1, 20)
    
    # Total belanja tahunan dipengaruhi oleh pendapatan, frekuensi, dan durasi keanggotaan
    belanja_base = (pendapatan_juta * 1.8) + (frekuensi_transaksi * 1.5) + (durasi_member_bulan * 0.4)
    total_belanja_juta = (belanja_base + np.random.normal(0, 3.5, n)).clip(5.0, 85.0).round(2)

    df = pd.DataFrame({
        "durasi_member_bulan": durasi_member_bulan,
        "pendapatan_juta": pendapatan_juta,
        "frekuensi_transaksi": frekuensi_transaksi,
        "total_belanja_juta": total_belanja_juta
    })
    return df


def run_eda_analysis(df: pd.DataFrame):
    """Menjalankan analisis statistik deskriptif dan visualisasi distribusi data."""
    print("=" * 65)
    print("   ANALISIS EKSPLORASI DATA (EDA) - CUSTOMER SPENDING ANALYTICS   ")
    print("=" * 65)
    print(f"Bentuk data: {df.shape[0]} baris, {df.shape[1]} kolom")
    
    print("\n1. 5 Sampel Data Teratas:")
    print(df.head().to_string())

    print("\n2. Ringkasan Statistik Deskriptif:")
    print(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())
    print("=" * 65)

    # Visualisasi Distribusi Total Belanja Pelanggan (Histogram & KDE)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["total_belanja_juta"], kde=True, color="#2b5c8f", bins=15, edgecolor="w")
    plt.title("Distribusi Frekuensi Total Belanja Pelanggan", fontsize=12, pad=12)
    plt.xlabel("Total Belanja Tahunan (Juta IDR)")
    plt.ylabel("Frekuensi Pelanggan")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_spending = create_customer_spending_dataset()
    run_eda_analysis(df_spending)
