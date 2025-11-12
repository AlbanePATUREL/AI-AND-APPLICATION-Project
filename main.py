import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============ Utils ============
def ensure_cols(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(missing)
            + "\nVérifie les en-têtes de ton fichier."
        )

def save_bar_labels(ax, fmt="{:.1f}%"):
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width()/2, h),
                    ha='center', va='bottom', fontsize=9)

def rate(series: pd.Series) -> float:
    """Retourne le pourcentage de 1."""
    return 100.0 * series.mean()

def cut_bins(series: pd.Series, edges: List[float], last_label_plus=False):
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        if i == len(edges) - 2 and last_label_plus:
            labels.append(f"{int(a)}+")
        else:
            if float(b).is_integer():
                labels.append(f"{int(a)}–{int(b)-1}")
            else:
                labels.append(f"{a}–{b}")
    # pour la dernière classe ouverte si last_label_plus
    bins = edges + ([10**9] if last_label_plus else [])
    return pd.cut(series, bins=bins, right=False, labels=labels)


# ============ Graphiques ============
def plot_overall(df, outdir):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    r = rate(df["Survived"])
    ax.bar(["Global"], [r])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie global")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_taux_global.png"), dpi=150)
    plt.close(fig)

def plot_by_sex(df, outdir):
    tmp = df.dropna(subset=["Sex"])
    if tmp.empty:
        return
    grp = 100 * tmp.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Sexe")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie par sexe")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_survie_par_sexe.png"), dpi=150)
    plt.close(fig)

def plot_by_agebins(df, outdir):
    tmp = df.dropna(subset=["Age"]).copy()
    if tmp.empty:
        return
    edges = [0,10,20,30,40,50,60,70,80]
    tmp["age_bin"] = cut_bins(tmp["Age"], edges, last_label_plus=True)
    grp = 100 * tmp.groupby("age_bin")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Âge")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie par tranche d'âge")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_survie_par_tranche_age.png"), dpi=150)
    plt.close(fig)

def plot_by_status(df, outdir):
    # Status = catégorie sociale (Citizen, Slave, Noble)
    tmp = df.dropna(subset=["Status"])
    if tmp.empty:
        return
    grp = 100 * tmp.groupby("Status")["Survived"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Statut social")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie par statut social")
    plt.xticks(rotation=15)
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_survie_par_statut.png"), dpi=150)
    plt.close(fig)

def plot_by_distance(df, outdir):
    # DistanceFromV = distance depuis le Vésuve (en km, supposé)
    tmp = df.dropna(subset=["DistanceFromV"]).copy()
    if tmp.empty:
        return
    # Bins réguliers (0–5–10–…); adaptons à l'étendue réelle
    mx = float(np.nanmax(tmp["DistanceFromV"]))
    step = 5.0 if mx > 25 else max(2.0, round(mx / 8, 1))
    edges = list(np.arange(0, np.ceil((mx+1e-9)/step)*step + step, step))
    if edges[-1] < mx:
        edges.append(mx)
    tmp["dist_bin"] = pd.cut(tmp["DistanceFromV"], bins=edges, right=False)
    grp = 100 * tmp.groupby("dist_bin")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([str(i) for i in grp.index], grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Distance depuis le Vésuve")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie selon la distance au volcan")
    plt.xticks(rotation=45, ha="right")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_survie_par_distance.png"), dpi=150)
    plt.close(fig)

def plot_by_wealth(df, outdir):
    tmp = df.dropna(subset=["WealthIndex"]).copy()
    if tmp.empty:
        return
    # Déciles de richesse
    tmp["wealth_decile"] = pd.qcut(tmp["WealthIndex"], q=10, labels=[f"D{i}" for i in range(1,11)])
    grp = 100 * tmp.groupby("wealth_decile")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Décile de richesse (WealthIndex)")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie par niveau de richesse (déciles)")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_survie_par_wealth_deciles.png"), dpi=150)
    plt.close(fig)

def plot_binary_feature(df, col, label, outdir, fname):
    tmp = df.dropna(subset=[col]).copy()
    if tmp.empty:
        return
    # normaliser 0/1 au cas où
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp[tmp[col].isin([0,1])]
    grp = 100 * tmp.groupby(col)["Survived"].mean().reindex([0,1])
    grp.index = ["Non", "Oui"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(grp.index.astype(str), grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel(label)
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title(f"Taux de survie selon {label.lower()}")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)

def plot_heatmap_sex_age(df, outdir):
    tmp = df.dropna(subset=["Sex", "Age"]).copy()
    if tmp.empty:
        return
    edges = [0,10,20,30,40,50,60,70,80]
    tmp["age_bin"] = cut_bins(tmp["Age"], edges, last_label_plus=True)
    pivot = tmp.pivot_table(values="Survived", index="Sex", columns="age_bin",
                            aggfunc="mean") * 100
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cax = ax.matshow(pivot.values, aspect="auto")
    ax.set_title("Carte thermique du taux de survie (Sexe × Âge)", pad=18)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=30, ha="left")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    cb = fig.colorbar(cax)
    cb.set_label("Taux de survie (%)")
    # annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "07_heatmap_sexe_age.png"), dpi=150)
    plt.close(fig)

def plot_reaction_time(df, outdir):
    tmp = df.dropna(subset=["ReactionTime"]).copy()
    if tmp.empty:
        return
    # Bins en quantiles pour homogénéiser les effectifs (ex: 8 quantiles)
    try:
        tmp["rt_bin"] = pd.qcut(tmp["ReactionTime"], q=8)
    except ValueError:
        # s'il y a trop de valeurs identiques, repli à 5 quantiles
        tmp["rt_bin"] = pd.qcut(tmp["ReactionTime"], q=5, duplicates="drop")
    grp = 100 * tmp.groupby("rt_bin")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([str(i) for i in grp.index], grp.values)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Temps de réaction (quantiles)")
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie selon le temps de réaction")
    plt.xticks(rotation=30, ha="right")
    save_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_survie_par_reaction_time.png"), dpi=150)
    plt.close(fig)


# ============ Pipeline principal ============
def main():
    parser = argparse.ArgumentParser(description="Analyse du taux de survie - Éruption du Vésuve (dataset ajusté)")
    parser.add_argument("--csv", required=True, help="Chemin du fichier CSV (vesuvius_survival_dataset.csv)")
    parser.add_argument("--sep", default=None, help="Séparateur (None = auto)")
    parser.add_argument("--encoding", default="utf-8", help="Encodage (utf-8 par défaut)")
    parser.add_argument("--out", default="figures", help="Dossier de sortie")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Lecture
    df = pd.read_csv(args.csv, sep=args.sep, encoding=args.encoding)
    # Normaliser les noms de colonnes en respectant TA table (sensible à la casse)
    df.columns = [c.strip() for c in df.columns]

    # Vérifier les colonnes essentielles
    required = ["Survived", "Sex", "Age", "Status",
                "DistanceFromV", "WealthIndex", "ShelterAccess", "HasPet", "ReactionTime"]
    ensure_cols(df, required)

    # Conversions utiles
    df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
    df = df[df["Survived"].isin([0, 1])]  # garder uniquement 0/1
    for col in ["Age", "DistanceFromV", "WealthIndex", "ReactionTime", "ShelterAccess", "HasPet"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Graphiques
    plot_overall(df, args.out)
    plot_by_sex(df, args.out)
    plot_by_agebins(df, args.out)
    plot_by_status(df, args.out)
    plot_by_distance(df, args.out)
    plot_by_wealth(df, args.out)
    plot_binary_feature(df, "ShelterAccess", "l'accès à un abri", args.out, "09_survie_selon_abri.png")
    plot_binary_feature(df, "HasPet", "le fait d'avoir un animal", args.out, "10_survie_selon_animal.png")
    plot_reaction_time(df, args.out)
    plot_heatmap_sex_age(df, args.out)

    # Récapitulatif texte + CSV
    recap = {}

    recap["Global_survival_%"] = round(rate(df["Survived"]), 2)

    by_sex = (100 * df.groupby("Sex")["Survived"].mean()).round(2)
    recap.update({f"Sex:{k}": v for k, v in by_sex.to_dict().items()})

    # tranches d'âge
    edges = [0,10,20,30,40,50,60,70,80]
    age_bin = cut_bins(df["Age"], edges, last_label_plus=True)
    by_age = (100 * df.groupby(age_bin)["Survived"].mean()).round(2)
    recap.update({f"Age:{str(k)}": v for k, v in by_age.dropna().to_dict().items()})

    by_status = (100 * df.groupby("Status")["Survived"].mean()).round(2)
    recap.update({f"Status:{k}": v for k, v in by_status.to_dict().items()})

    # Export recap
    recap_series = pd.Series(recap, name="value")
    recap_series.to_csv(os.path.join(args.out, "00_recap_taux.csv"))
    print("=== RÉSUMÉ ===")
    print(recap_series)

    print(f"\nFigures et récap enregistrés dans : {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
