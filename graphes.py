#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_survival_rate(survival_rate, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / "01_overall_survival_rate.png"
    fig = plt.figure()
    plt.bar(["Survival rate"], [survival_rate])
    plt.ylim(0, 1)
    plt.ylabel("Rate (0-1)")
    plt.title("Overall Survival Rate")
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")

def plot_survival_by_age_tens(df, surv_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    s = pd.to_numeric(df[surv_col], errors="coerce")
    s = (s > 0).astype(int)
    age = pd.to_numeric(df["Age"], errors="coerce")
    data = pd.DataFrame({"Age": age, "Survived": s}).dropna()
    if data.empty:
        raise ValueError("No valid data for Age/Survived.")
    data["age_decade"] = (data["Age"] // 10).astype(int) * 10
    grp = data.groupby("age_decade")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    fig = plt.figure()
    plt.plot(range(len(grp)), grp.values, marker="o")
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Age (decades)")
    plt.title("Survival rate by age (per decade)")
    out_file = out_dir / "02_survival_by_age_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")

def plot_survival_by_distance_tens(df, surv_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    s = pd.to_numeric(df[surv_col], errors="coerce")
    s = (s > 0).astype(int)
    dist = pd.to_numeric(df["DistanceFromV"], errors="coerce")
    data = pd.DataFrame({"DistanceFromV": dist, "Survived": s}).dropna()
    if data.empty:
        raise ValueError("No valid data for DistanceFromV/Survived.")
    data = data[data["DistanceFromV"] >= 0]
    data["dist_decade"] = (data["DistanceFromV"] // 10).astype(int) * 10
    grp = data.groupby("dist_decade")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    fig = plt.figure()
    plt.plot(range(len(grp)), grp.values, marker="o")
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Distance from Vesuvius (per decade)")
    plt.title("Survival rate by distance (per decade)")
    out_file = out_dir / "03_survival_by_distance_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")

def plot_survival_by_gender(df, surv_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if "Sex" not in df.columns and "Gender" not in df.columns:
        raise KeyError("No 'Sex' or 'Gender' column found in dataset.")

    gender_col = "Sex" if "Sex" in df.columns else "Gender"
    s = pd.to_numeric(df[surv_col], errors="coerce")
    s = (s > 0).astype(int)
    data = pd.DataFrame({"Gender": df[gender_col], "Survived": s}).dropna()

    grouped = data.groupby("Gender")["Survived"].mean()
    fig = plt.figure()
    plt.bar(grouped.index, grouped.values)
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Gender")
    plt.title("Survival rate by gender")
    out_file = out_dir / "04_survival_by_gender.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")

def plot_survival_by_reaction_time(df, surv_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if "ReactionTime" not in df.columns:
        raise KeyError("No 'ReactionTime' column found in dataset.")
    s = pd.to_numeric(df[surv_col], errors="coerce")
    s = (s > 0).astype(int)
    reaction = pd.to_numeric(df["ReactionTime"], errors="coerce")
    data = pd.DataFrame({"ReactionTime": reaction, "Survived": s}).dropna()
    if data.empty:
        raise ValueError("No valid data for ReactionTime/Survived.")
    data["reaction_bin"] = (data["ReactionTime"] // 10).astype(int) * 10
    grp = data.groupby("reaction_bin")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    fig = plt.figure()
    plt.plot(range(len(grp)), grp.values, marker="o")
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Reaction time (by tens)")
    plt.title("Survival rate by reaction time")
    out_file = out_dir / "05_survival_by_reaction_time.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")

def plot_survival_by_status(df, surv_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if "Status" not in df.columns:
        raise KeyError("No 'Status' column found in dataset.")
    s = pd.to_numeric(df[surv_col], errors="coerce")
    s = (s > 0).astype(int)
    data = pd.DataFrame({"Status": df["Status"], "Survived": s}).dropna()
    if data.empty:
        raise ValueError("No valid data for Status/Survived.")
    grouped = data.groupby("Status")["Survived"].mean().sort_values(ascending=False)
    fig = plt.figure()
    plt.bar(grouped.index, grouped.values)
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Status")
    plt.title("Survival rate by status")
    plt.xticks(rotation=30, ha="right")
    out_file = out_dir / "06_survival_by_status.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"Image saved: {out_file.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="vesuvius_survival_dataset.csv")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, sep=args.sep, encoding=args.encoding)
    candidates = ["Survived", "survived", "Survival", "Outcome", "is_survived"]
    surv_col = next((c for c in candidates if c in df.columns), None)
    if surv_col is None:
        raise KeyError(f"No survival column found. Available columns: {list(df.columns)}")

    s = pd.to_numeric(df[surv_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError("No usable data found in the survival column.")
    s = (s > 0).astype(int)

    survival_rate = s.mean()
    print(f"Overall survival rate: {survival_rate:.3f}")

    plot_survival_rate(survival_rate, Path(args.out))
    plot_survival_by_age_tens(df, surv_col, args.out)
    plot_survival_by_distance_tens(df, surv_col, args.out)
    plot_survival_by_gender(df, surv_col, args.out)
    plot_survival_by_reaction_time(df, surv_col, args.out)
    plot_survival_by_status(df, surv_col, args.out)

if __name__ == "__main__":
    main()
