# src/eda/eda_tools.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

sns.set(style="whitegrid", rc={"figure.dpi": 150})

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- Summaries ----------
def data_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtypes and non-null counts."""
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null_count": df.count(),
        "null_count": df.isna().sum(),
        "unique": df.nunique(dropna=False)
    })
    return info

def descriptive_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df[cols].describe().T

# ---------- Business Metric: Loss Ratio ----------
def overall_loss_ratio(df: pd.DataFrame) -> float:
    total_claims = df["TotalClaims"].sum(skipna=True)
    total_premium = df["TotalPremium"].sum(skipna=True)
    if total_premium == 0:
        return np.nan
    return total_claims / total_premium

def loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grp = df.groupby(group_col)[["TotalPremium","TotalClaims"]].sum()
    grp = grp.assign(LossRatio = grp["TotalClaims"] / grp["TotalPremium"])
    grp = grp.sort_values("LossRatio", ascending=False)
    return grp

# ---------- Time series ----------
def monthly_claims_premiums(df: pd.DataFrame, date_col: str = "TransactionMonth") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    monthly = df.groupby(pd.Grouper(key=date_col, freq="MS"))[["TotalClaims","TotalPremium"]].sum()
    monthly["ClaimFrequency"] = df.groupby(pd.Grouper(key=date_col, freq="MS"))["TotalClaims"].apply(lambda s: (s>0).sum())
    # severity: average claim amount per claim (avoid div by zero)
    monthly["ClaimSeverity"] = monthly.apply(lambda r: r["TotalClaims"] / max(r["ClaimFrequency"], 1), axis=1)
    return monthly

# ---------- Outlier detection ----------
def outlier_summary(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper,
            "n_outliers": ((s < lower) | (s > upper)).sum()}

# ---------- Plots (3 required polished plots) ----------
def plot_loss_ratio_by_province(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    grp = loss_ratio_by_group(df, "Province")
    plt.figure(figsize=(10,6))
    sns.barplot(x=grp.index, y=grp["LossRatio"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Loss Ratio (TotalClaims / TotalPremium)")
    plt.title("Loss Ratio by Province")
    plt.tight_layout()
    path = os.path.join(outdir, "loss_ratio_by_province.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_totalclaims_distribution(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    plt.figure(figsize=(8,5))
    # log scale helps when heavy skew/outliers
    sns.histplot(df["TotalClaims"].dropna(), bins=100, kde=True)
    plt.xscale('symlog')  # symmetric log to keep zeros visible
    plt.xlabel("TotalClaims (symlog scale)")
    plt.title("Distribution of TotalClaims (log-friendly)")
    plt.tight_layout()
    path = os.path.join(outdir, "totalclaims_distribution.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_claims_premium_time_series(df: pd.DataFrame, outdir: str, date_col="TransactionMonth"):
    ensure_dir(outdir)
    monthly = monthly_claims_premiums(df, date_col=date_col)
    plt.figure(figsize=(10,6))
    ax = monthly[["TotalClaims","TotalPremium"]].plot(title="Monthly TotalClaims vs TotalPremium")
    ax.set_ylabel("Amount (local currency)")
    plt.tight_layout()
    path = os.path.join(outdir, "monthly_claims_premium.png")
    plt.savefig(path)
    plt.close()
    return path

# ---------- Bivariate exploration ----------
def scatter_premium_vs_claims(df: pd.DataFrame, outdir: str, sample=10000):
    ensure_dir(outdir)
    n = min(len(df), sample)
    sample_df = df.sample(n=n, random_state=42)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=sample_df["TotalPremium"], y=sample_df["TotalClaims"], alpha=0.6)
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.xlabel("TotalPremium (symlog)")
    plt.ylabel("TotalClaims (symlog)")
    plt.title(f"Scatter: TotalPremium vs TotalClaims (sample n={n})")
    plt.tight_layout()
    path = os.path.join(outdir, "scatter_premium_vs_claims.png")
    plt.savefig(path)
    plt.close()
    return path
