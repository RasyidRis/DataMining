"""
Supervised Learning: Prediksi Retensi & Churn Pelanggan (Klasifikasi)
Memprediksi potensi churn pelanggan e-commerce menggunakan Random Forest Classifier.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def build_ecommerce_churn_dataset() -> pd.DataFrame:
    """Membangkitkan data perilaku transaksi pelanggan digital untuk prediksi retensi."""
    np.random.seed(42)
    n = 250

    hari_terakhir_aktif = np.random.randint(1, 90, n)
    frekuensi_belanja = np.random.poisson(lam=7.0, size=n).clip(1, 35)
    usia_akun_bulan = np.random.randint(1, 36, n)
    tiket_komplain_cs = np.random.choice([0, 1, 2, 3, 5], size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05])

    # Logit probabilitas churn berbasis recency, frekuensi belanja, dan keluhan
    logit = (0.07 * hari_terakhir_aktif) - (0.15 * frekuensi_belanja) + (0.65 * tiket_komplain_cs) - (0.04 * usia_akun_bulan) - 2.2
    prob_churn = 1 / (1 + np.exp(-logit))
    status_churn = (np.random.rand(n) < prob_churn).astype(int)

    df = pd.DataFrame({
        "hari_terakhir_aktif": hari_terakhir_aktif,
        "frekuensi_belanja": frekuensi_belanja,
        "usia_akun_bulan": usia_akun_bulan,
        "tiket_komplain_cs": tiket_komplain_cs,
        "status_churn": status_churn
    })
    return df


def train_churn_model(df: pd.DataFrame):
    """Melatih Random Forest Classifier dan mengevaluasi performa prediksi serta feature importance."""
    feature_cols = ["hari_terakhir_aktif", "frekuensi_belanja", "usia_akun_bulan", "tiket_komplain_cs"]
    X = df[feature_cols]
    y = df["status_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.24, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    prediksi = rf_model.predict(X_test)
    probabilitas = rf_model.predict_proba(X_test)[:, 1]
    skor_roc = roc_auc_score(y_test, probabilitas)

    print("=== LAPORAN EVALUASI PREDIKSI CUSTOMER CHURN ===")
    print("Metrik Kinerja Model (Classification Report):")
    print(classification_report(y_test, prediksi, target_names=["Aktif (0)", "Churn (1)"]))
    print(f"Skor ROC-AUC: {skor_roc:.4f}")

    print("\nFaktor Paling Berpengaruh Terhadap Churn (Feature Importance):")
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for rank, (col, val) in enumerate(importances.items(), start=1):
        print(f"{rank}. {col:<22} : {val:.4f}")


if __name__ == "__main__":
    df_churn = build_ecommerce_churn_dataset()
    train_churn_model(df_churn)
