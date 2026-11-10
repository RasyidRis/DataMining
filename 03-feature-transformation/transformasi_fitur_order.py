"""
Modul: 03-feature-transformation
Topik: Categorical Encoding, Feature Scaling, dan Feature Creation
Studi Kasus: Rekayasa Fitur Transaksi & Pengiriman E-Commerce
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def build_sample_order_data() -> pd.DataFrame:
    """Membuat data sampel transaksi pesanan untuk demonstrasi rekayasa fitur."""
    return pd.DataFrame({
        "wilayah": ["Jawa", "Sumatera", "Bali", "Jawa", "Kalimantan", "Sumatera", "Sulawesi", "Jawa"],
        "layanan_kirim": ["Reguler", "Express", "Kargo", "Reguler", "Express", "Reguler", "Kargo", "Express"],
        "total_belanja_ribu": [150, 450, 1200, 280, 850, 320, 2400, 600],
        "jumlah_item": [2, 5, 12, 4, 10, 3, 25, 6],
        "durasi_kirim_hari": [2, 1, 4, 2, 3, 3, 5, 1],
    })


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline transformasi dan rekayasa fitur baru."""
    df_mod = df.copy()

    # 1. Feature Creation: Rata-rata nominal belanja per item barang
    df_mod["belanja_per_item"] = (df_mod["total_belanja_ribu"] / df_mod["jumlah_item"]).round(2)

    # 2. Log Transformation: Meredam kemiringan (skewness) nilai uang belanja
    df_mod["log_total_belanja"] = np.log1p(df_mod["total_belanja_ribu"]).round(2)

    # 3. Column Transformer: One-Hot Encoding untuk kategori & Min-Max Scaling untuk numerik
    kolom_kategori = ["wilayah", "layanan_kirim"]
    kolom_numerik = ["durasi_kirim_hari", "belanja_per_item", "log_total_belanja"]

    transformer = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), kolom_numerik),
            ("cat", OneHotEncoder(sparse_output=False, drop="first"), kolom_kategori),
        ],
        remainder="drop",
    )

    matriks_hasil = transformer.fit_transform(df_mod)
    nama_fitur = transformer.get_feature_names_out()

    df_hasil = pd.DataFrame(matriks_hasil, columns=nama_fitur).round(3)
    return df_hasil


if __name__ == "__main__":
    print("=== EKSEKUSI PIPELINE FEATURE TRANSFORMATION ===")
    data_awal = build_sample_order_data()
    print("1. Data Awal:")
    print(data_awal.to_string())

    data_rekayasa = apply_feature_engineering(data_awal)
    print("\n2. Data Hasil Transformasi Fitur (Encoding & Scaling):")
    print(data_rekayasa.to_string())
