"""
Modul: 01-data-cleaning
Topik: Advanced Data Cleaning, Outlier Handling, and Imputation Pipeline
Studi Kasus: E-Commerce Customer Activity & Transaction Dataset
"""

import numpy as np
import pandas as pd

try:
    from sklearn.impute import KNNImputer, SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_raw_ecommerce_dataset() -> pd.DataFrame:
    """Membuat synthetic raw dataset transaksi e-commerce."""
    np.random.seed(42)
    n = 120

    order_ids = [f"ORD-2026-{1000 + i}" for i in range(n)]

    age_choices = [21, 26, 31, 38, 45, 52, np.nan, -7, 140]
    age_probs = [0.20, 0.25, 0.20, 0.15, 0.08, 0.04, 0.04, 0.02, 0.02]
    customer_age = np.random.choice(age_choices, size=n, p=age_probs)

    tier_choices = [
        "Bronze", "bronze", "BRONZE", 
        "Silver", "silver ", " SILVER",
        "Gold", "gold", "GOLD ", 
        "Platinum", "None", "?", None
    ]
    tier_probs = [0.12, 0.10, 0.08, 0.15, 0.10, 0.08, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.03]
    membership_tier = np.random.choice(tier_choices, size=n, p=tier_probs)

    spend_base = np.random.exponential(scale=350, size=n) + 50
    spend_base[np.random.choice(n, size=4, replace=False)] = [8500, 12000, 15500, -250]
    spend_base[np.random.choice(n, size=6, replace=False)] = np.nan
    total_spend = np.round(spend_base, 2)

    items_base = np.random.poisson(lam=3.5, size=n)
    items_base[np.random.choice(n, size=3, replace=False)] = 0
    items_count = items_base.astype(float)
    items_count[np.random.choice(n, size=5, replace=False)] = np.nan

    dist_base = np.random.gamma(shape=2.5, scale=4.0, size=n)
    dist_base[np.random.choice(n, size=5, replace=False)] = np.nan
    delivery_distance = np.round(dist_base, 2)

    rating_choices = [1.0, 2.0, 3.0, 4.0, 5.0, 9.5, -1.0, np.nan]
    rating_probs = [0.05, 0.10, 0.25, 0.35, 0.17, 0.02, 0.02, 0.04]
    customer_rating = np.random.choice(rating_choices, size=n, p=rating_probs)

    df = pd.DataFrame({
        "order_id": order_ids,
        "customer_age": customer_age,
        "membership_tier": membership_tier,
        "total_spend_k": total_spend,
        "items_count": items_count,
        "delivery_distance_km": delivery_distance,
        "customer_rating": customer_rating
    })

    duplicate_rows = df.iloc[[3, 17, 45, 78]].copy()
    df_raw = pd.concat([df, duplicate_rows], ignore_index=True)
    return df_raw


def clean_ecommerce_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline pembersihan data transaksi."""
    df_clean = df.copy()

    # 1. Deduplikasi
    df_clean = df_clean.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)

    # 2. Standardisasi Teks & Kategori
    df_clean["membership_tier"] = df_clean["membership_tier"].astype(str).str.strip().str.title()
    sentinels = ["None", "Nan", "?", "Unknown", "Null", ""]
    df_clean["membership_tier"] = df_clean["membership_tier"].replace(sentinels, np.nan)
    valid_tiers = {"Bronze", "Silver", "Gold", "Platinum"}
    df_clean.loc[~df_clean["membership_tier"].isin(valid_tiers) & df_clean["membership_tier"].notna(), "membership_tier"] = np.nan

    # 3. Validasi Batasan Nilai Logis Domain
    df_clean.loc[(df_clean["customer_age"] < 17) | (df_clean["customer_age"] > 85), "customer_age"] = np.nan
    df_clean.loc[df_clean["total_spend_k"] <= 0, "total_spend_k"] = np.nan
    df_clean.loc[df_clean["items_count"] < 1, "items_count"] = np.nan
    df_clean.loc[df_clean["delivery_distance_km"] <= 0, "delivery_distance_km"] = np.nan
    df_clean.loc[(df_clean["customer_rating"] < 1.0) | (df_clean["customer_rating"] > 5.0), "customer_rating"] = np.nan

    # 4. Outlier Capping (IQR Method)
    for col in ["total_spend_k", "delivery_distance_km"]:
        valid_series = df_clean[col].dropna()
        q1 = valid_series.quantile(0.25)
        q3 = valid_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - (1.5 * iqr))
        upper_bound = q3 + (1.5 * iqr)
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # 5. Imputasi Nilai Kosong
    tier_mode = df_clean["membership_tier"].mode()[0] if not df_clean["membership_tier"].mode().empty else "Bronze"
    df_clean["membership_tier"] = df_clean["membership_tier"].fillna(tier_mode)

    numeric_cols = ["customer_age", "total_spend_k", "items_count", "delivery_distance_km", "customer_rating"]
    if SKLEARN_AVAILABLE:
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    else:
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean["customer_age"] = df_clean["customer_age"].round().astype(int)
    df_clean["items_count"] = df_clean["items_count"].round().astype(int)
    df_clean["customer_rating"] = df_clean["customer_rating"].round(1)
    df_clean["total_spend_k"] = df_clean["total_spend_k"].round(2)
    df_clean["delivery_distance_km"] = df_clean["delivery_distance_km"].round(2)

    return df_clean


def print_summary_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Mencetak laporan evaluasi kualitas data (Data Quality Report)."""
    print("=" * 65)
    print("      LAPORAN EVALUASI KUALITAS DATA (DATA QUALITY REPORT)      ")
    print("=" * 65)
    print(f"1. Jumlah Baris Data:")
    print(f"   - Mentah (Raw)      : {len(df_raw)} baris")
    print(f"   - Hasil Bersih      : {len(df_clean)} baris (Deduplikasi: {len(df_raw) - len(df_clean)} baris)")
    
    print("\n2. Perbandingan Missing Values (Nilai Kosong):")
    missing_comparison = pd.DataFrame({
        "Missing (Raw)": df_raw.isnull().sum(),
        "Missing (Clean)": df_clean.isnull().sum()
    })
    print(missing_comparison.to_string())

    print("\n3. Distribusi Kategori Membership (Clean):")
    print(df_clean["membership_tier"].value_counts().to_string())

    print("\n4. Statistik Deskriptif Numerik (Setelah Capping & Imputasi):")
    stats = df_clean[["customer_age", "total_spend_k", "items_count", "customer_rating"]].describe().T
    print(stats[["mean", "std", "min", "50%", "max"]].to_string())

    print("\n5. Cuplikan 5 Baris Data Terbersih:")
    print(df_clean.head().to_string())
    print("=" * 65)


if __name__ == "__main__":
    print("Memulai eksekusi pipeline data preprocessing...")
    raw_data = create_raw_ecommerce_dataset()
    cleaned_data = clean_ecommerce_pipeline(raw_data)
    print_summary_report(raw_data, cleaned_data)
