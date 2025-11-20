#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vesuvius Eruption Survival Analysis
Uses Random Forest for prediction and generates comprehensive visualizations
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
    """Load and display data overview"""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")
    
    data = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    
    print("="*70)
    print("DATA OVERVIEW")
    print("="*70)
    print("\nFirst rows:")
    print(data.head())
    print("\nDataset information:")
    print(data.info())
    print("\nDescriptive statistics:")
    print(data.describe())
    
    return data


def prepare_data(data):
    """Prepare data for training"""
    # Copy to avoid modifying original
    data_clean = data.drop(['PassengerId', 'Name'], axis=1, errors='ignore')
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = [col for col in ['Sex', 'Status', 'Gender'] if col in data_clean.columns]
    
    for col in categorical_columns:
        le = LabelEncoder()
        data_clean[col] = le.fit_transform(data_clean[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = data_clean.drop(['Survived'], axis=1)
    y = data_clean['Survived']
    
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    print(f"Features used: {X.columns.tolist()}")
    
    return X, y, label_encoders


def train_model(X, y):
    """Train Random Forest model"""
    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining dataset size: {len(X_train)}")
    print(f"Test dataset size: {len(X_test)}")
    print(f"Class distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
    
    # Model training
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
    """Evaluate model and display metrics"""
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("MODEL RESULTS")
    print("="*70)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Deceased', 'Survivor']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Error analysis
    errors = y_test != y_pred
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Number of errors: {errors.sum()} out of {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")
    
    return y_pred, y_pred_proba


def predict_full_dataset(rf_model, scaler, X, y, data):
    """Generate predictions for entire dataset"""
    X_full_scaled = scaler.transform(X)
    predictions_full = rf_model.predict(X_full_scaled)
    predictions_proba_full = rf_model.predict_proba(X_full_scaled)
    
    print(f"\n{'='*70}")
    print(f"PREDICTION ON FULL DATASET")
    print(f"{'='*70}")
    print(f"\nPredicted overall survival rate: {predictions_full.mean()*100:.2f}%")
    print(f"Actual survival rate: {y.mean()*100:.2f}%")
    
    return predictions_full, predictions_proba_full


def save_predictions(data, predictions_full, output_file="vesuvius_survival_predictions.csv"):
    """Save predictions to CSV file"""
    output_df = pd.DataFrame({
        'PassengerId': data['PassengerId'],
        'Survived': predictions_full
    })
    
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to '{output_file}'")
    print(f"✓ Format: PassengerId,Survived")
    print(f"✓ Total records: {len(output_df)}")


def create_visualizations(data, X, y, rf_model, scaler, X_test, y_test, 
                         y_pred, y_pred_proba, out_dir="plots"):
    """Create all visualizations"""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for charts - encoding for correlation
    data_clean = data.drop(['PassengerId', 'Name'], axis=1, errors='ignore').copy()
    
    # Encode categorical columns for correlation
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
    print("FEATURE IMPORTANCE")
    print("="*70)
    print(feature_importance)
    
    # Figure 1: 4 main model charts
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Survival probability vs Distance
    if 'DistanceFromV' in X.columns:
        distance_idx = list(X.columns).index('DistanceFromV')
        distances_test = X_test[:, distance_idx] * scaler.scale_[distance_idx] + scaler.mean_[distance_idx]
        survival_proba_test = y_pred_proba[:, 1]
        
        sort_idx = np.argsort(distances_test)
        distances_sorted = distances_test[sort_idx]
        proba_sorted = survival_proba_test[sort_idx]
        
        axes[0, 0].scatter(distances_test[y_test == 0], survival_proba_test[y_test == 0], 
                           alpha=0.6, c='red', label='Actually Deceased', s=50, edgecolors='darkred')
        axes[0, 0].scatter(distances_test[y_test == 1], survival_proba_test[y_test == 1], 
                           alpha=0.6, c='green', label='Actually Survived', s=50, edgecolors='darkgreen')
        
        window = 10
        if len(distances_sorted) >= window:
            proba_smooth = uniform_filter1d(proba_sorted, size=window)
            axes[0, 0].plot(distances_sorted, proba_smooth, 'b-', linewidth=2.5, 
                            label='Trend', alpha=0.8)
        
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Threshold')
        axes[0, 0].set_title('Survival Probability vs Distance from Vesuvius', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Distance from Vesuvius (km)')
        axes[0, 0].set_ylabel('Survival Probability')
        axes[0, 0].legend(loc='best')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(-0.05, 1.05)
    
    # 2. Feature importance
    feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Importance', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Importance')
    
    # 3. Correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 0], vmin=-1, vmax=1, 
                cbar_kws={'label': 'Correlation coefficient'})
    axes[1, 0].set_title('Correlation Matrix between Variables', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0)
    
    # 4. Actual vs Predicted comparison
    axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Actual', s=30)
    axes[1, 1].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted', s=30, marker='x')
    axes[1, 1].set_title('Comparison: Actual vs Predicted Survival', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Sample index')
    axes[1, 1].set_ylabel('Survival (0=No, 1=Yes)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(out_path / "00_model_analysis.png", dpi=140, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Chart saved: {out_path / '00_model_analysis.png'}")
    
    # Individual charts by factor
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
    """Chart of overall survival rate"""
    survival_rate = y.mean()
    out_file = out_dir / "01_overall_survival_rate.png"
    
    fig = plt.figure(figsize=(8, 6))
    plt.bar(["Survival rate"], [survival_rate], color='steelblue', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Rate (0-1)")
    plt.title(f"Overall Survival Rate: {survival_rate:.1%}", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def plot_survival_by_age_tens(data, out_dir):
    """Survival rate by age decade"""
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
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Age (decades)")
    plt.title("Survival Rate by Age", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "02_survival_by_age_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def plot_survival_by_distance_tens(data, out_dir):
    """Survival rate by distance from Vesuvius"""
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
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Distance from Vesuvius (by decade)")
    plt.title("Survival Rate by Distance from Vesuvius", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "03_survival_by_distance_tens.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def plot_survival_by_gender(data, out_dir):
    """Survival rate by gender"""
    gender_col = "Sex" if "Sex" in data.columns else "Gender"
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Gender": data[gender_col], "Survived": survived}).dropna()
    
    grouped = df.groupby("Gender")["Survived"].mean()
    
    fig = plt.figure(figsize=(8, 6))
    plt.bar(grouped.index, grouped.values, color=['lightcoral', 'lightblue'], alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Gender")
    plt.title("Survival Rate by Gender", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    out_file = out_dir / "04_survival_by_gender.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def plot_survival_by_reaction_time(data, out_dir):
    """Survival rate by reaction time"""
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
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Reaction time (by tens)")
    plt.title("Survival Rate by Reaction Time", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    out_file = out_dir / "05_survival_by_reaction_time.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def plot_survival_by_status(data, out_dir):
    """Survival rate by social status"""
    survived = pd.to_numeric(data["Survived"], errors="coerce")
    df = pd.DataFrame({"Status": data["Status"], "Survived": survived}).dropna()
    
    if df.empty:
        return
    
    grouped = df.groupby("Status")["Survived"].mean().sort_values(ascending=False)
    
    fig = plt.figure(figsize=(10, 6))
    plt.bar(grouped.index, grouped.values, color='mediumpurple', alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Survival rate (0-1)")
    plt.xlabel("Social status")
    plt.title("Survival Rate by Social Status", fontweight='bold')
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis='y', alpha=0.3)
    
    out_file = out_dir / "06_survival_by_status.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=140)
    plt.show()
    print(f"✓ Chart saved: {out_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Vesuvius eruption survival analysis with Random Forest"
    )
    parser.add_argument("--csv", default="vesuvius_survival_dataset.csv",
                       help="Path to CSV file")
    parser.add_argument("--sep", default=",", help="CSV separator")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    parser.add_argument("--out", default="plots", help="Output folder for charts")
    parser.add_argument("--predictions", default="vesuvius_survival_predictions.csv",
                       help="Output file for predictions")
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv)
    data = load_and_explore_data(csv_path, args.sep, args.encoding)
    
    # Prepare data
    X, y, label_encoders = prepare_data(data)
    
    # Train model
    rf_model, scaler, X_train, X_test, y_train, y_test, X_scaled = train_model(X, y)
    
    # Evaluation
    y_pred, y_pred_proba = evaluate_model(rf_model, X_test, y_test)
    
    # Full predictions
    predictions_full, predictions_proba_full = predict_full_dataset(
        rf_model, scaler, X, y, data
    )
    
    # Save predictions
    save_predictions(data, predictions_full, args.predictions)
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    create_visualizations(data, X, y, rf_model, scaler, X_test, y_test,
                         y_pred, y_pred_proba, args.out)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"✓ Predictions: {args.predictions}")
    print(f"✓ Charts: {args.out}/")


if __name__ == "__main__":
    main()
