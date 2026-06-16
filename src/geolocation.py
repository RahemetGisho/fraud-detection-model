"""
Geolocation Integration
Responsibilities:
  1. Ensure ip_address is stored as a proper integer 
  2. Merge Fraud_Data with IpAddress_to_Country using a range-based lookup
  3. Analyse fraud patterns by country (count + rate).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_DEFAULT_PLOT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "plots"
)

def _ensure(plot_dir: str) -> str:
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def merge_ip_to_country(fraud_df: pd.DataFrame,
                         ip_df: pd.DataFrame) -> pd.DataFrame:
    if fraud_df["ip_address"].dtype != np.int64:
        fraud_df = fraud_df.copy()
        fraud_df["ip_address"] = fraud_df["ip_address"].astype(np.int64)
    ip_work = ip_df.copy()
    ip_work["lower_bound_ip_address"] = (
        ip_work["lower_bound_ip_address"].astype(np.int64)
    )
    ip_work["upper_bound_ip_address"] = (
        ip_work["upper_bound_ip_address"].astype(np.int64)
    )
    fraud_sorted = fraud_df.sort_values("ip_address").reset_index(drop=False)
    ip_sorted    = ip_work.sort_values("lower_bound_ip_address")
    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    in_range = merged["ip_address"] <= merged["upper_bound_ip_address"]
    merged.loc[~in_range, "country"] = "Unknown"
    merged.drop(
        columns=["lower_bound_ip_address", "upper_bound_ip_address"],
        inplace=True,
        errors="ignore",
    )
    merged.sort_values("index", inplace=True)
    merged.drop(columns=["index"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    n_unknown = (~in_range).sum()
    pct_unknown = n_unknown / len(merged) * 100
    print(f"[geolocation] Mapped {len(merged):,} transactions to countries.")
    print(f"[geolocation] Unmapped IPs (→ 'Unknown'): "
          f"{n_unknown:,} ({pct_unknown:.1f}%)")
    print(f"[geolocation] Unique countries found: "
          f"{merged['country'].nunique()}")
    return merged

def fraud_rate_by_country(df: pd.DataFrame,
                           target_col: str = "class",
                           min_transactions: int = 10,
                           top_n: int = 15,
                           plot_dir: str = _DEFAULT_PLOT_DIR) -> pd.DataFrame:
    _ensure(plot_dir)

    stats = (
        df.groupby("country")[target_col]
        .agg(total="count", fraud="sum")
        .reset_index()
    )
    stats["fraud_rate"] = stats["fraud"] / stats["total"]
    stats = stats[stats["total"] >= min_transactions]
    print(f"\n[geolocation] Countries with ≥ {min_transactions} transactions: "
          f"{len(stats)}")

    top_count = stats.nlargest(top_n, "fraud")
    top_rate  = stats.nlargest(top_n, "fraud_rate")
    print(f"\nTop {top_n} by fraud count:\n"
          + top_count[["country", "total", "fraud", "fraud_rate"]].to_string(
              index=False))
    print(f"\nTop {top_n} by fraud rate:\n"
          + top_rate[["country", "total", "fraud", "fraud_rate"]].to_string(
              index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    top_count.set_index("country")["fraud"].sort_values().plot(
        kind="barh", ax=ax1, color="steelblue", alpha=0.85
    )
    ax1.set_title(f"Top {top_n} Countries — Absolute Fraud Count", fontsize=11)
    ax1.set_xlabel("Fraud transactions")

    top_rate.set_index("country")["fraud_rate"].sort_values().plot(
        kind="barh", ax=ax2, color="tomato", alpha=0.85
    )
    ax2.set_title(f"Top {top_n} Countries — Fraud Rate", fontsize=11)
    ax2.set_xlabel("Fraud rate")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    out = os.path.join(plot_dir, "fraud_by_country.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"[geolocation] Saved → {out}")

    return stats.sort_values("fraud_rate", ascending=False).reset_index(
        drop=True
    )