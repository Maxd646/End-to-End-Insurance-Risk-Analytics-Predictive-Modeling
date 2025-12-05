# scripts/run_eda.py
import os
from src.utils.data_loader import load_csv, save_csv
from src.eda.eda_tools import (
    data_structure, descriptive_stats, overall_loss_ratio, loss_ratio_by_group,
    plot_loss_ratio_by_province, plot_totalclaims_distribution,
    plot_claims_premium_time_series, scatter_premium_vs_claims, outlier_summary
)
import argparse
import json

def main(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    print("Loading data:", input_path)
    df = load_csv(input_path, parse_dates=["TransactionMonth", "VehicleIntroDate"])

    # Basic data structure
    structure = data_structure(df)
    structure.to_csv(os.path.join(summaries_dir, "data_structure.csv"))

    # Descriptive stats for key numeric columns
    numeric_cols = ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
    present = [c for c in numeric_cols if c in df.columns]
    stats = descriptive_stats(df, present)
    stats.to_csv(os.path.join(summaries_dir, "descriptive_stats.csv"))

    # Compute Loss Ratios
    overall_lr = overall_loss_ratio(df)
    lr_by_province = loss_ratio_by_group(df, "Province").reset_index()
    lr_by_province.to_csv(os.path.join(summaries_dir, "loss_ratio_by_province.csv"))
    with open(os.path.join(summaries_dir, "loss_ratio_overall.json"), "w") as f:
        json.dump({"overall_loss_ratio": overall_lr}, f, default=str)

    # Outliers summary
    outlier_tc = outlier_summary(df, "TotalClaims") if "TotalClaims" in df.columns else {}
    with open(os.path.join(summaries_dir, "outlier_totalclaims.json"), "w") as f:
        json.dump(outlier_tc, f, default=str)

    # Create required 3 beautiful plots
    p1 = plot_loss_ratio_by_province(df, figures_dir)
    p2 = plot_totalclaims_distribution(df, figures_dir)
    p3 = plot_claims_premium_time_series(df, figures_dir)
    p4 = scatter_premium_vs_claims(df, figures_dir)

    print("Saved figures:", p1, p2, p3, p4)
    print("Summaries saved to", summaries_dir)
    print("Overall Loss Ratio:", overall_lr)
    print("EDA complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA for ACIS dataset")
    parser.add_argument("--input", default="data/raw/data.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    main(args.input, args.output)
