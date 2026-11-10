"""
Supervised Learning: Estimasi Harga Laptop (Regresi)
Eksperimen prediksi harga laptop berdasarkan spesifikasi perangkat keras menggunakan Linear Regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def build_laptop_dataset() -> pd.DataFrame:
    """Membangkitkan data sintetis spesifikasi teknis dan harga laptop."""
    np.random.seed(42)
    n = 180

    ram_gb = np.random.choice([4, 8, 16, 32], size=n, p=[0.15, 0.45, 0.30, 0.10])
    storage_ssd_gb = np.random.choice([256, 512, 1024], size=n, p=[0.30, 0.50, 0.20])
    skor_prosesor = np.random.randint(60, 220, size=n)

    # Rumus linier harga: basis spesifikasi + variasi acak
    harga = (ram_gb * 0.45) + (storage_ssd_gb * 0.008) + (skor_prosesor * 0.065) + np.random.normal(0, 1.2, n) + 2.5
    harga_juta = np.round(np.clip(harga, 3.8, 38.0), 2)

    df = pd.DataFrame({
        "ram_gb": ram_gb,
        "storage_ssd_gb": storage_ssd_gb,
        "skor_prosesor": skor_prosesor,
        "harga_juta": harga_juta
    })
    return df


def train_laptop_price_model(df: pd.DataFrame):
    """Melatih model regresi linier dan mengevaluasi metrik performa."""
    X = df[["ram_gb", "storage_ssd_gb", "skor_prosesor"]]
    y = df["harga_juta"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)
    r2 = r2_score(y_test, prediksi)
    mae = mean_absolute_error(y_test, prediksi)
    rmse = np.sqrt(mean_squared_error(y_test, prediksi))

    print("=== EVALUASI MODEL REGRESI HARGA LAPTOP ===")
    print(f"Jumlah data latih : {len(X_train)} baris")
    print(f"Jumlah data uji   : {len(X_test)} baris")
    print(f"R-Squared (R2)    : {r2:.4f}")
    print(f"MAE               : Rp {mae:.2f} Juta")
    print(f"RMSE              : Rp {rmse:.2f} Juta")


if __name__ == "__main__":
    data_laptop = build_laptop_dataset()
    train_laptop_price_model(data_laptop)
