#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de survie lors de l'éruption du Vésuve
Utilise Random Forest pour la prédiction et génère des visualisations complètes
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.ndimage import uniform_filter1d


def load_and_explore_data(csv_path, sep=",", encoding="utf-8"):
    """Charge et affiche un aperçu des données"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path.resolve()}")
    
    data = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    
    print("="*70)
    print("APERÇU DES DONNÉES")
    print("="*70)
    print("\nPremières lignes:")
    print(data.head())
    print("\nInformations sur le dataset:")
    print(data.info())
    print("\nStatistiques descriptives:")
    print(data.describe())
    
    return data


def prepare_data(data):
    """Prépare les données pour l'entraînement"""
    # Copie pour éviter de modifier l'original
    data_clean = data.drop(['PassengerId', 'Name'], axis=1, errors='ignore')
    
    # Encodage des variables catégorielles
    label_encoders = {}
    categorical_columns = [col for col in ['Sex', 'Status', 'Gender'] if col in data_clean.columns]
    
    for col in categorical_columns:
        le = LabelEncoder()
        data_clean[col] = le.fit_transform(data_clean[col])
        label_encoders[col] = le
    
    # Séparation features et target
    X = data_clean.drop(['Survived'], axis=1)
    y = data_clean['Survived']
    
    print("\n" + "="*70)
    print("PRÉPARATION DES DONNÉES")
    print("="*70)
    print(f"Features utilisées: {X.columns.tolist()}")
    
    return X, y, label_encoders


def train_model(X, y):
    """Entraîne le modèle Random Forest"""
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTaille du dataset d'entraînement: {len(X_train)}")
    print(f"Taille du dataset de test: {len(X_test)}")
    print(f"Distribution des classes (train): {np.bincount(y_train)}")
    print(f"Distribution des classes (test): {np.bincount(y_test)}")
    
    # Entraînement du modèle
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    return rf_model, scaler, X_train, X_test, y_train, y_test, X_scaled


def evaluate_model(rf_model, X_test, y_test):
    """Évalue le modèle et affiche les métriques"""
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("RÉSULTATS DU MODÈLE")
    print("="*70)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Décédé', 'Survivant']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion:")
    print(cm)
    
    # Analyse des erreurs
    errors = y_test != y_pred
    print(f"\n{'='*70}")
    print(f"ANALYSE DES ERREURS")
    print(f"{'='*70}")
    print(f"Nombre d'erreurs: {errors.sum()} sur {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")
    
    return y_pred, y_pred_proba


def predict_full_dataset(rf_model, scaler, X, y, data):
    """Génère les prédictions pour l'ensemble du dataset"""
    X_full_scaled = scaler.transform(X)
    predictions_full = rf_model.predict(X_full_scaled)
    predictions_proba_full = rf_model.predict_proba(X_full_scaled)
    
    print(f"\n{'='*70}")
    print(f"PRÉDICTION SUR L'ENSEMBLE DU DATASET")
    print(f"{'='*70}")
    print(f"\nTaux de survie prédit global: {predictions_full.mean()*100:.2f}%")
    print(f"Taux de survie réel: {y.mean()*100:.2f}%")
    
    return predictions_full, predictions_proba_full


def save_predictions(data, predictions_full, output_file="vesuvius_survival_predictions.csv"):
    """Sauvegarde les prédictions dans un fichier CSV"""
    output_df = pd.DataFrame({
        'PassengerId': data['PassengerId'],
        'Survived': predictions_full
    })
    
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Résultats sauvegardés dans '{output_file}'")
    print(f"✓ Format: PassengerId,Survived")
    print(f"✓ Total d'enregistrements: {len(output_df)}")


def create_visualizations(data, X, y, rf_model, scaler, X_test, y_test, 
                         y_pred, y_pred_proba, out_dir="plots"):
    """Crée toutes les visualisations"""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Préparation des données pour les graphiques - encodage pour corrélation
    data_clean = data.drop(['PassengerId', 'Name'], axis=1, errors='ignore').copy()
    
    # Encoder les colonnes catégorielles pour la corrélation
    for col in data_clean.select_dtypes(include=['object']).columns:
        if col in data_clean.columns:
            le = LabelEncoder()
            data_clean[col] = le.fit_transform(data_clean[col])
    
    correlation_matrix = data_clean.corr(numeric_only=True)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*70)
    print("IMPORTANCE DES FEATURES")
    print("="*70)
    print(feature_importance)
    
    # Figure 1: 4 graphiques principaux du modèle
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Probabilité de survie vs Distance
    if 'DistanceFromV' in X.columns:
        distance_idx = list(X.columns).index('DistanceFromV')
        distances_test = X_test[:, distance_idx] * scaler.scale_[distance_idx] + scaler.mean_[distance_idx]
        survival_proba_test = y_pred_proba[:, 1]
        
        sort_idx = np.argsort(distances_test)
        distances_sorted = distances_test[sort_idx]
        proba_sorted = survival_proba_test[sort_idx]
        
        axes[0, 0].scatter(distances_test[y_test == 0], survival_proba_test[y_test == 0], 
                           alpha=0.6, c='red', label='Réellement Décédés', s=50, edgecolors='darkred')
        axes[0, 0].scatter(distances_test[y_test == 1], survival_proba_test[y_test == 1], 
                           alpha=0.6, c='green', label='Réellement Survivants', s=50, edgecolors='darkgreen')
        
        window = 10
        if len(distances_sorted) >= window:
            proba_smooth = uniform_filter1d(proba_sorted, size=window)
            axes[0, 0].plot(distances_sorted, proba_smooth, 'b-', linewidth=2.5, 
                            label='Tendance', alpha=0.8)
        
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Seuil 50%')
        axes[0, 0].set_title('Probabilité de Survie vs Distance du Vésuve', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Distance du Vésuve (km)')
        axes[0, 0].set_ylabel('Probabilité de Survie')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-0.05, 1.05)
    
    # 2. Importance des features
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1])
    axes[0, 1].set_title('Importance des Features', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Matrice de corrélation
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 0], vmin=-1, vmax=1, 
                cbar_kws={'label': 'Coefficient de corrélation'})
    axes[1, 0].set_title('Matrice de Corrélation entre les Variables', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0)
    
    # 4. Comparaison Réel vs Prédit
    axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Réel', s=30)
    axes[1, 1].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Prédit', s=30, marker='x')
    axes[1, 1].set_title('Comparaison: Survie Réelle vs Prédite', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Index de l\'échantillon')
    axes[1, 1].set_ylabel('Survie (0=Non, 1=Oui)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(out_path / "00_model_analysis.png", dpi=140, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Graphique sauvegardé: {out_path / '00_model_analysis.png'}")
    
    # Graphiques individuels par facteur
    plot_survival_rate(y, out_path)
    plot_survival_by_age_tens(data, out_path)
    plot_survival_by_distance_tens(data, out_path)
    
    if 'Sex' in data.columns or 'Gender' in data.columns:
        plot_survival_by_gender(data, out_path)
    
    if 'ReactionTime' in data.columns:
        plot_survival_by_reaction_time(data, out_path)
    
    if 'Status' in data.columns:
        plot_survival_by_status(data, out_path)


def plot_survival_rate(y, out_dir):
    """Graphique du taux de survie global"""
    survival_rate = y.mean()
    out_file = out_dir / "01_overall_survival_rate.png"
    
    fig = plt.figure(figsize=(8, 6))
    plt.bar(["Taux de survie"], [survival_rate], color='steelblue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Taux (0-1)")
    plt.title(f"Taux de Survie Global: {survival_rate:.1%}", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def plot_survival_by_age_tens(data, out_dir):
    """Taux de survie par décennie d'âge"""
    age = pd.to_numeric(data["Age"], errors="coerce")
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Age": age, "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    df["age_decade"] = (df["Age"] // 10).astype(int) * 10
    grp = df.groupby("age_decade")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(grp)), grp.values, marker="o", linewidth=2, markersize=8, color='darkblue')
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Taux de survie (0-1)")
    plt.xlabel("Âge (décennies)")
    plt.title("Taux de Survie par Âge", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "02_survival_by_age_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def plot_survival_by_distance_tens(data, out_dir):
    """Taux de survie par distance du Vésuve"""
    dist = pd.to_numeric(data["DistanceFromV"], errors="coerce")
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"DistanceFromV": dist, "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    df = df[df["DistanceFromV"] >= 0]
    df["dist_decade"] = (df["DistanceFromV"] // 10).astype(int) * 10
    grp = df.groupby("dist_decade")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9} km" for d in grp.index]
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(grp)), grp.values, marker="o", linewidth=2, markersize=8, color='darkgreen')
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Taux de survie (0-1)")
    plt.xlabel("Distance du Vésuve (par décennie)")
    plt.title("Taux de Survie par Distance du Vésuve", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "03_survival_by_distance_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def plot_survival_by_gender(data, out_dir):
    """Taux de survie par sexe"""
    gender_col = "Sex" if "Sex" in data.columns else "Gender"
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Gender": data[gender_col], "Survived": survived}).dropna()
    
    grouped = df.groupby("Gender")["Survived"].mean()
    
    fig = plt.figure(figsize=(8, 6))
    plt.bar(grouped.index, grouped.values, color=['lightcoral', 'lightblue'], alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Taux de survie (0-1)")
    plt.xlabel("Sexe")
    plt.title("Taux de Survie par Sexe", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    out_file = out_dir / "04_survival_by_gender.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def plot_survival_by_reaction_time(data, out_dir):
    """Taux de survie par temps de réaction"""
    reaction = pd.to_numeric(data["ReactionTime"], errors="coerce")
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"ReactionTime": reaction, "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    df["reaction_bin"] = (df["ReactionTime"] // 10).astype(int) * 10
    grp = df.groupby("reaction_bin")["Survived"].mean().sort_index()
    labels = [f"{int(d)}-{int(d)+9}" for d in grp.index]
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(grp)), grp.values, marker="o", linewidth=2, markersize=8, color='darkorange')
    plt.xticks(range(len(grp)), labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Taux de survie (0-1)")
    plt.xlabel("Temps de réaction (par dizaines)")
    plt.title("Taux de Survie par Temps de Réaction", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "05_survival_by_reaction_time.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def plot_survival_by_status(data, out_dir):
    """Taux de survie par statut social"""
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Status": data["Status"], "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    grouped = df.groupby("Status")["Survived"].mean().sort_values(ascending=False)
    
    fig = plt.figure(figsize=(10, 6))
    plt.bar(grouped.index, grouped.values, color='mediumpurple', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Taux de survie (0-1)")
    plt.xlabel("Statut social")
    plt.title("Taux de Survie par Statut Social", fontweight='bold')
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis='y', alpha=0.3)
    
    out_file = out_dir / "06_survival_by_status.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Graphique sauvegardé: {out_file}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Analyse de survie lors de l'éruption du Vésuve avec Random Forest"
    )
    parser.add_argument("--csv", default="vesuvius_survival_dataset.csv",
                       help="Chemin vers le fichier CSV")
    parser.add_argument("--sep", default=",", help="Séparateur CSV")
    parser.add_argument("--encoding", default="utf-8", help="Encodage du fichier")
    parser.add_argument("--out", default="plots", help="Dossier de sortie pour les graphiques")
    parser.add_argument("--predictions", default="vesuvius_survival_predictions.csv",
                       help="Fichier de sortie pour les prédictions")
    args = parser.parse_args()
    
    # Chargement des données
    csv_path = Path(args.csv)
    data = load_and_explore_data(csv_path, args.sep, args.encoding)
    
    # Préparation des données
    X, y, label_encoders = prepare_data(data)
    
    # Entraînement du modèle
    rf_model, scaler, X_train, X_test, y_train, y_test, X_scaled = train_model(X, y)
    
    # Évaluation
    y_pred, y_pred_proba = evaluate_model(rf_model, X_test, y_test)
    
    # Prédictions complètes
    predictions_full, predictions_proba_full = predict_full_dataset(
        rf_model, scaler, X, y, data
    )
    
    # Sauvegarde des prédictions
    save_predictions(data, predictions_full, args.predictions)
    
    # Visualisations
    print("\n" + "="*70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*70)
    create_visualizations(data, X, y, rf_model, scaler, X_test, y_test,
                         y_pred, y_pred_proba, args.out)
    
    print("\n" + "="*70)
    print("✓ ANALYSE TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"✓ Prédictions: {args.predictions}")
    print(f"✓ Graphiques: {args.out}/")


if __name__ == "__main__":
    main()