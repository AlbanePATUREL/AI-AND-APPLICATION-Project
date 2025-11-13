import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# Chargement des données
data = pd.read_csv('vesuvius_survival_dataset.csv')

print("Aperçu des données:")
print(data.head())
print("\nInformations sur le dataset:")
print(data.info())
print("\nStatistiques descriptives:")
print(data.describe())

# Préparation des données
# Suppression de la colonne PassengerId et Name (non pertinentes pour la prédiction)
data_clean = data.drop(['PassengerId', 'Name'], axis=1)

# Encodage des variables catégorielles
label_encoders = {}
categorical_columns = ['Sex', 'Status']

for col in categorical_columns:
    le = LabelEncoder()
    data_clean[col] = le.fit_transform(data_clean[col])
    label_encoders[col] = le

# Séparation des features et de la target
X = data_clean.drop(['Survived'], axis=1)
y = data_clean['Survived']

print("\nFeatures utilisées:", X.columns.tolist())

# Normalisation des features
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

# Création et entraînement du modèle Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Prédictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"RÉSULTATS DU MODÈLE")
print(f"{'='*50}")
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred, 
                          target_names=['Décédé', 'Survivant']))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion:")
print(cm)

# Importance des features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportance des features:")
print(feature_importance)

# Calcul de la matrice de corrélation
correlation_matrix = data_clean.corr()
print("\nMatrice de corrélation:")
print(correlation_matrix)

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Récupérer l'index de la colonne DistanceFromV pour tous les graphiques
distance_idx = list(X.columns).index('DistanceFromV')

# Extraire les distances du test set (dénormaliser)
distances_test = X_test[:, distance_idx] * scaler.scale_[distance_idx] + scaler.mean_[distance_idx]
survival_proba_test = y_pred_proba[:, 1]

# 1. Probabilité de Survie en fonction de la Distance (Vue d'ensemble)
# Trier par distance pour une meilleure visualisation
sort_idx = np.argsort(distances_test)
distances_sorted = distances_test[sort_idx]
proba_sorted = survival_proba_test[sort_idx]

# Scatter plot avec les données réelles
axes[0, 0].scatter(distances_test[y_test == 0], survival_proba_test[y_test == 0], 
                   alpha=0.6, c='red', label='Réellement Décédés', s=50, edgecolors='darkred')
axes[0, 0].scatter(distances_test[y_test == 1], survival_proba_test[y_test == 1], 
                   alpha=0.6, c='green', label='Réellement Survivants', s=50, edgecolors='darkgreen')

# Ajouter une ligne de tendance (moyenne mobile)
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

# 3. Matrice de corrélation (remplace la matrice de confusion)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 0], vmin=-1, vmax=1, 
            cbar_kws={'label': 'Coefficient de corrélation'})
axes[1, 0].set_title('Matrice de Corrélation entre les Variables', fontweight='bold', fontsize=12)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0)

# 4. Comparaison des facteurs clés
feature_comparison = pd.DataFrame({
    'DistanceFromV': X_test[:, 0],
    'Survived_Real': y_test,
    'Survived_Pred': y_pred
})

axes[1, 1].scatter(range(len(y_test)), y_test, alpha=0.5, label='Réel', s=30)
axes[1, 1].scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Prédit', s=30, marker='x')
axes[1, 1].set_title('Comparaison: Survie Réelle vs Prédite')
axes[1, 1].set_xlabel('Index de l\'échantillon')
axes[1, 1].set_ylabel('Survie (0=Non, 1=Oui)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyse des erreurs
errors = y_test != y_pred
print(f"\n{'='*50}")
print(f"ANALYSE DES ERREURS")
print(f"{'='*50}")
print(f"Nombre d'erreurs: {errors.sum()} sur {len(y_test)} ({errors.sum()/len(y_test)*100:.2f}%)")

# Prédiction sur l'ensemble du dataset
print(f"\n{'='*50}")
print(f"PRÉDICTION SUR L'ENSEMBLE DU DATASET")
print(f"{'='*50}")

X_full_scaled = scaler.transform(X)
predictions_full = rf_model.predict(X_full_scaled)
predictions_proba_full = rf_model.predict_proba(X_full_scaled)

print(f"\nTaux de survie prédit global: {predictions_full.mean()*100:.2f}%")
print(f"Taux de survie réel: {y.mean()*100:.2f}%")

# Création du fichier de sortie avec seulement PassengerId et Survived
output_df = pd.DataFrame({
    'PassengerId': data['PassengerId'],
    'Survived': predictions_full
})

# Sauvegarde des résultats
output_df.to_csv('vesuvius_survival_predictions.csv', index=False)
print("\nRésultats sauvegardés dans 'vesuvius_survival_predictions.csv'")
print(f"Format du fichier: PassengerId, Survived")
print(f"Nombre de lignes: {len(output_df)}")

# Afficher quelques exemples de prédictions
print("\n" + "="*50)
print("EXEMPLES DE PRÉDICTIONS")
print("="*50)

# Créer un DataFrame avec plus d'informations pour l'affichage
display_df = pd.DataFrame({
    'PassengerId': data['PassengerId'],
    'Name': data['Name'],
    'DistanceFromV': data['DistanceFromV'],
    'Age': data['Age'],
    'Status': data['Status'],
    'Survived_Real': data['Survived'],
    'Survived_Predicted': predictions_full,
    'Probability': predictions_proba_full[:, 1]
})

print(display_df.head(15))
print(f"\n✓ Fichier de sortie créé: vesuvius_survival_predictions.csv")
print(f"✓ Format: PassengerId,Survived")
print(f"✓ Total d'enregistrements: {len(output_df)}")

# 1. Se déplacer dans le répertoire contenant vos fichiers
#cd "C:\chemin\vers\votre\dossier"

# 2. Vérifier que les fichiers nécessaires sont présents
#ls vesuvius_survival_dataset.csv

# 3. Créer un nouvel environnement Conda avec Python
#conda create -n pompeii python=3.11 -y

# 4. Activer l'environnement Conda
#conda activate pompeii

# 5. Installer les packages nécessaires avec Conda
#conda install pandas numpy scikit-learn matplotlib seaborn -y

# Alternative :cd installer avec pip dans l'environnement Conda
# pip install pandas numpy scikit-learn matplotlib seaborn

# 6. Vérifier les packages installés
#conda list

# 7. Créer le fichier Python avec le code
# Copiez le code dans un fichier nommé "pompeii_survival.py"

# 8. Exécuter le script
#python pompeii_survival.py
