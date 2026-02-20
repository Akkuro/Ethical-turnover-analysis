"""Generate the complete main.ipynb notebook for HumanForYou Attrition Analysis."""

import json


def md(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = source.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source[:-1]] + [source[-1]],
    }


def code(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source[:-1]] + [source[-1]],
    }


cells = []

# =============================================================================
# TITLE
# =============================================================================
cells.append(
    md("""# Analyse Éthique du Turnover : HumanForYou
## Pipeline complète : EDA → Régression → Classification

**Objectif** : Identifier les facteurs de départ des employés et proposer des leviers d'action RH.

**Dataset** : 5 fichiers CSV (general_data, employee_survey, manager_survey, in_time, out_time)""")
)

# =============================================================================
# ETHICS SECTION
# =============================================================================
cells.append(
    md("""# Cadrage Éthique

Avant toute modélisation, il est essentiel d'identifier les variables susceptibles de mener à du **profilage discriminatoire**. L'objectif n'est pas de prédire *qui* va partir (profilage individuel), mais d'identifier les **leviers organisationnels** sur lesquels l'entreprise peut agir.

## Variables exclues de la modélisation

| Variable | Type de risque | Justification du retrait |
|---|---|---|
| **Gender** | Critère protégé (discrimination directe) | Code du travail L.1132-1, RGPD art. 9. Un modèle qui corrèle genre et attrition normaliserait un biais structurel. |
| **Age** | Critère protégé (discrimination directe) | Un modèle qui apprend « les jeunes partent plus » pourrait conduire à discriminer à l'embauche. |
| **MaritalStatus** | Critère protégé (situation familiale) | Corrélé au genre ; pénalise la vie privée de l'employé. |
| **avg_work_hours** | Surveillance / Profilage comportemental | Issu du pointage individuel (in_time/out_time). Risque de normaliser le présentéisme. |
| **std_work_hours** | Surveillance / Profilage comportemental | Même source. Pénalise les horaires atypiques (temps partiel thérapeutique, parents…). |
| **days_absent** | Surveillance + Discrimination indirecte | Corrélé aux congés maladie, maternité, handicap : tous protégés par la loi. Art. 22 RGPD (profilage automatisé). |
| **DistanceFromHome** | Non actionnable | L'entreprise ne peut pas modifier le lieu de résidence de ses employés. Conserver cette variable pousserait à discriminer à l'embauche selon la localisation géographique du candidat. |

> **Note sur Over18** : cette colonne est **constante** (tous les employés ont la valeur `Y`). Une colonne constante a une variance nulle : elle n'apporte aucune information discriminante au modèle et est donc supprimée pour raison technique, indépendamment de sa nature éthique.

## Variables conservées avec vigilance

| Variable | Risque résiduel | Raison de conservation |
|---|---|---|
| **EducationField** | Proxy potentiel du genre (filières genrées) | Conservé car levier RH (plans de formation ciblés), mais à surveiller via audit de disparate impact. |
| **MonthlyIncome** | Reflète des inégalités historiques | Conservé en régression (cible) et comme levier salarial ; ne pas l'utiliser pour justifier des écarts existants. |

## Principe directeur

> **Les prédictions du modèle ne doivent jamais être appliquées à un individu.**  
> Elles servent à détecter des **tendances organisationnelles** (politique salariale, fréquence des promotions, charge de travail) afin d'améliorer les conditions de travail pour tous.""")
)

# =============================================================================
# PART 1 : EDA
# =============================================================================
cells.append(
    md("""# Partie 1 : Analyse Exploratoire des Données (EDA)

> Inspiré du Workshop « Utilisation IA : Exploratory Data Analysis »""")
)

cells.append(md("## 1.1 Environnement et imports"))

cells.append(
    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

%matplotlib inline

print("Environnement prêt ✓")""")
)

cells.append(md("## 1.2 Chargement des données"))

cells.append(
    code("""# Chargement des 5 fichiers CSV
general = pd.read_csv('../data/general_data.csv')
emp_survey = pd.read_csv('../data/employee_survey_data.csv')
mgr_survey = pd.read_csv('../data/manager_survey_data.csv')
in_time = pd.read_csv('../data/in_time.csv')
out_time = pd.read_csv('../data/out_time.csv')

print(f"general_data      : {general.shape}")
print(f"employee_survey   : {emp_survey.shape}")
print(f"manager_survey    : {mgr_survey.shape}")
print(f"in_time           : {in_time.shape}")
print(f"out_time          : {out_time.shape}")""")
)

cells.append(md("## 1.3 Exploration initiale"))

cells.append(
    code("""# Aperçu de chaque table
for name, df in [('general', general), ('emp_survey', emp_survey),
                  ('mgr_survey', mgr_survey), ('in_time', in_time), ('out_time', out_time)]:
    print(f"\\n{'='*60}")
    print(f" {name} : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"{'='*60}")
    print(df.dtypes.value_counts().to_string())
    print(f"\\nValeurs manquantes : {df.isnull().sum().sum()}")""")
)

cells.append(code("""general.head()"""))

cells.append(code("""general.describe(include='all').T"""))

cells.append(
    md("""## 1.4 Feature Engineering : Heures de travail

Les fichiers `in_time` et `out_time` contiennent les horodatages d'entrée/sortie pour chaque jour ouvré de 2015.  
On va en extraire des métriques agrégées par employé :
- **avg_work_hours** : nombre moyen d'heures travaillées par jour
- **std_work_hours** : écart-type des heures travaillées (régularité)
- **days_absent** : nombre de jours d'absence (NaN dans in_time)""")
)

cells.append(
    code("""# Renommer la première colonne (non nommée) en EmployeeID
in_time = in_time.rename(columns={in_time.columns[0]: 'EmployeeID'})
out_time = out_time.rename(columns={out_time.columns[0]: 'EmployeeID'})

# Convertir les colonnes de dates en datetime
date_cols = in_time.columns[1:]

in_time_dt = in_time.copy()
out_time_dt = out_time.copy()
for col in date_cols:
    in_time_dt[col] = pd.to_datetime(in_time_dt[col], errors='coerce')
    out_time_dt[col] = pd.to_datetime(out_time_dt[col], errors='coerce')

# Calcul des heures travaillées par jour
work_hours = pd.DataFrame()
work_hours['EmployeeID'] = in_time_dt['EmployeeID']

for col in date_cols:
    diff = (out_time_dt[col] - in_time_dt[col]).dt.total_seconds() / 3600
    work_hours[col] = diff

# Agrégation par employé
time_features = pd.DataFrame()
time_features['EmployeeID'] = work_hours['EmployeeID']
time_features['avg_work_hours'] = work_hours[date_cols].mean(axis=1)
time_features['std_work_hours'] = work_hours[date_cols].std(axis=1)
time_features['days_absent'] = work_hours[date_cols].isnull().sum(axis=1)

print(time_features.describe())
time_features.head()""")
)

cells.append(
    md("""## 1.5 Fusion des datasets

On fusionne toutes les tables sur `EmployeeID` pour obtenir un dataset unique.""")
)

cells.append(
    code("""# Fusion progressive sur EmployeeID
df = general.merge(emp_survey, on='EmployeeID', how='left')
df = df.merge(mgr_survey, on='EmployeeID', how='left')
df = df.merge(time_features, on='EmployeeID', how='left')

print(f"Dataset fusionné : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
df.head()""")
)

cells.append(md("""## 1.6 Analyse des valeurs manquantes"""))

cells.append(
    code("""# Matrice de valeurs manquantes
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

msno.matrix(df, ax=axes[0], sparkline=False, fontsize=8)
axes[0].set_title('Matrice des valeurs manquantes', fontsize=14)

# Barplot des valeurs manquantes
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Colonnes avec valeurs manquantes')
    axes[1].set_xlabel('Nombre de NaN')
    for i, v in enumerate(missing):
        axes[1].text(v + 0.5, i, f'{v} ({v/len(df)*100:.1f}%)', va='center')
else:
    axes[1].text(0.5, 0.5, 'Aucune valeur manquante !', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=14)
plt.tight_layout()
plt.show()""")
)

cells.append(
    md("""## 1.7 Nettoyage des données

1. **Encodage** de la variable cible `Attrition` en binaire (Yes=1, No=0)
2. **Suppression** des colonnes constantes (EmployeeCount, StandardHours, Over18)
3. **Suppression** de EmployeeID (identifiant, pas une feature)
4. **Suppression éthique** des variables sensibles / protégées (Gender, Age, MaritalStatus)
5. **Suppression éthique** des métriques de surveillance (avg_work_hours, std_work_hours, days_absent)
6. **Suppression pragmatique** de DistanceFromHome (non actionnable par l'entreprise)
6. **Imputation** des valeurs manquantes numériques par la médiane""")
)

cells.append(
    code("""# Encoder Attrition en binaire
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Identifier les colonnes constantes
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Colonnes constantes : {constant_cols}")

# --- Colonnes à supprimer ---
# 1) Techniques : constantes + identifiant
cols_to_drop = constant_cols + ['EmployeeID']
for c in ['Over18', 'StandardHours', 'EmployeeCount']:
    if c in df.columns and c not in cols_to_drop:
        cols_to_drop.append(c)

# 2) Éthiques : critères protégés (discrimination directe)
ethical_protected = ['Gender', 'Age', 'MaritalStatus']

# 3) Éthiques : métriques de surveillance issues du pointage
ethical_surveillance = ['avg_work_hours', 'std_work_hours', 'days_absent']

# 4) Pragmatique : variable non actionnable par l'entreprise
non_actionable = ['DistanceFromHome']

cols_to_drop_ethical = [c for c in ethical_protected + ethical_surveillance + non_actionable if c in df.columns]

print(f"\\n--- Suppressions techniques ---")
print(f"Colonnes constantes / inutiles : {[c for c in cols_to_drop if c in df.columns]}")
print(f"\\n--- Suppressions éthiques ---")
print(f"Critères protégés (loi anti-discrimination) : {[c for c in ethical_protected if c in df.columns]}")
print(f"Surveillance comportementale (RGPD art. 22) : {[c for c in ethical_surveillance if c in df.columns]}")
print(f"\\n--- Suppressions pragmatiques ---")
print(f"Non actionnable par l'entreprise : {[c for c in non_actionable if c in df.columns]}")

all_to_drop = list(set(cols_to_drop + cols_to_drop_ethical))
df = df.drop(columns=[c for c in all_to_drop if c in df.columns])

print(f"\\nTotal colonnes supprimées : {len([c for c in all_to_drop if c in df.columns or c in all_to_drop])}")
print(f"Dataset après nettoyage : {df.shape}")""")
)

cells.append(
    code("""# Imputation des valeurs manquantes numériques par la médiane
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        count = df[col].isnull().sum()
        df[col].fillna(median_val, inplace=True)
        print(f"{col}: {count} NaN remplacés par médiane = {median_val}")

# Vérification
print(f"\\nValeurs manquantes restantes : {df.isnull().sum().sum()}")""")
)

cells.append(md("""## 1.8 Analyse univariée"""))

cells.append(md("""### Distribution de la variable cible (Attrition)"""))

cells.append(
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Countplot
counts = df['Attrition'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['No (0)', 'Yes (1)'], counts.values, color=colors, edgecolor='black')
axes[0].set_title('Distribution de Attrition', fontsize=14)
axes[0].set_ylabel('Nombre d\\'employés')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

# Pie chart
axes[1].pie(counts.values, labels=['No', 'Yes'], autopct='%1.1f%%',
            colors=colors, startangle=90, explode=[0, 0.05])
axes[1].set_title('Proportion Attrition', fontsize=14)

plt.suptitle(f'Variable cible : Attrition (n={len(df)})', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print(f"\\n⚠️ Déséquilibre de classes : {counts[0]/counts[1]:.1f}:1 (No:Yes)")""")
)

cells.append(md("### Distribution des variables numériques"))

cells.append(
    code("""num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols_no_target = [c for c in num_cols if c != 'Attrition']

n_cols = 4
n_rows = (len(num_cols_no_target) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols_no_target):
    sns.histplot(df[col], ax=axes[i], kde=True, bins=30, color='steelblue')
    axes[i].set_title(col, fontsize=11)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', alpha=0.7, label=f'mean={df[col].mean():.1f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', alpha=0.7, label=f'median={df[col].median():.1f}')
    axes[i].legend(fontsize=7)

# Masquer les axes inutilisés
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution des variables numériques', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()""")
)

cells.append(md("### Distribution des variables catégorielles"))

cells.append(
    code("""cat_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(cat_cols) > 0:
    n_cols_plot = 3
    n_rows_plot = (len(cat_cols) + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 5 * n_rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, ax=axes[i], order=order, palette='Set2')
        axes[i].set_title(col, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution des variables catégorielles', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("Aucune variable catégorielle restante.")""")
)

cells.append(md("""## 1.9 Analyse bivariée"""))

cells.append(md("### Matrice de corrélation"))

cells.append(
    code("""# Matrice de corrélation
plt.figure(figsize=(18, 14))
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 7})
plt.title('Matrice de corrélation', fontsize=16)
plt.tight_layout()
plt.show()""")
)

cells.append(md("### Top corrélations avec Attrition"))

cells.append(
    code("""# Corrélations avec la cible
corr_attrition = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)

plt.figure(figsize=(10, 8))
colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr_attrition.values]
corr_attrition.plot(kind='barh', color=colors)
plt.title('Corrélation avec Attrition', fontsize=14)
plt.xlabel('Coefficient de corrélation de Pearson')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\\nTop 10 corrélations (valeur absolue) :")
print(corr_attrition.head(10).to_string())""")
)

cells.append(md("### Boxplots : Variables numériques vs Attrition"))

cells.append(
    code("""# Top variables numériques les plus corrélées avec Attrition
top_num = corr_attrition.head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(top_num):
    sns.boxplot(data=df, x='Attrition', y=col, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xticklabels(['No', 'Yes'])

plt.suptitle('Variables numériques vs Attrition (top 8 corrélations)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("### Variables catégorielles vs Attrition"))

cells.append(
    code("""if len(cat_cols) > 0:
    n_cols_plot = 3
    n_rows_plot = (len(cat_cols) + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 5 * n_rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        # Taux d'attrition par catégorie
        attrition_rate = df.groupby(col)['Attrition'].mean().sort_values(ascending=False)
        attrition_rate.plot(kind='bar', ax=axes[i], color='coral', edgecolor='black')
        axes[i].set_title(f'Taux d\\'attrition par {col}', fontsize=12)
        axes[i].set_ylabel('Taux d\\'attrition')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].axhline(y=df['Attrition'].mean(), color='red', linestyle='--', alpha=0.5,
                        label=f'Moyenne = {df["Attrition"].mean():.2f}')
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Taux d\\'attrition par variable catégorielle', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()""")
)

cells.append(
    md("""## 1.10 Synthèse EDA

**Observations clés** :
- Le dataset est **déséquilibré** (~84% No vs ~16% Yes) → nécessite `class_weight='balanced'` en classification
- Les variables les plus corrélées avec l'attrition sont typiquement : TotalWorkingYears, YearsAtCompany, Age, MonthlyIncome (négativement), et avg_work_hours
- Les features issues de in_time/out_time apportent de l'information supplémentaire (heures travaillées, régularité)
- Peu de valeurs manquantes dans l'ensemble

---""")
)

# =============================================================================
# PART 2 : REGRESSION
# =============================================================================
cells.append(
    md("""# Partie 2 : Régression

> Inspiré du Workshop « Régression »

**Objectif** : Prédire le **MonthlyIncome** des employés à partir de leurs caractéristiques.  
On compare plusieurs approches : équation normale, statsmodels, sklearn (LinearRegression, DecisionTree, RandomForest, GradientBoosting, KNN, SVR).""")
)

cells.append(md("## 2.1 Préparation des données pour la régression"))

cells.append(
    code("""from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Variable cible pour la régression
target_reg = 'MonthlyIncome'

# Séparer features et cible
X_reg = df.drop(columns=[target_reg])
y_reg = df[target_reg].copy()

# Identifier les types de colonnes
num_features = X_reg.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_reg.select_dtypes(include=['object']).columns.tolist()

print(f"Features numériques ({len(num_features)}) : {num_features}")
print(f"Features catégorielles ({len(cat_features)}) : {cat_features}")
print(f"Cible : {target_reg}")
print(f"Shape X: {X_reg.shape}, Shape y: {y_reg.shape}")""")
)

cells.append(
    code("""# Pipelines de prétraitement
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# Split train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Prétraiter
X_train_processed = preprocessor.fit_transform(X_train_reg)
X_test_processed = preprocessor.transform(X_test_reg)

print(f"X_train transformé : {X_train_processed.shape}")
print(f"X_test transformé  : {X_test_processed.shape}")""")
)

cells.append(
    md("""## 2.2 Équation normale (OLS)

La solution analytique de la régression linéaire : $\\hat{\\theta} = (X^T X)^{-1} X^T y$""")
)

cells.append(
    code("""# Ajout du biais (colonne de 1)
X_b_train = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]
X_b_test = np.c_[np.ones((X_test_processed.shape[0], 1)), X_test_processed]

# Équation normale
theta_best = np.linalg.pinv(X_b_train.T @ X_b_train) @ X_b_train.T @ y_train_reg.values

# Prédictions
y_pred_normal = X_b_test @ theta_best

# Métriques
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse_normal = mean_squared_error(y_test_reg, y_pred_normal)
rmse_normal = np.sqrt(mse_normal)
r2_normal = r2_score(y_test_reg, y_pred_normal)
mae_normal = mean_absolute_error(y_test_reg, y_pred_normal)

print("=== Équation Normale ===")
print(f"MSE  : {mse_normal:,.0f}")
print(f"RMSE : {rmse_normal:,.0f}")
print(f"MAE  : {mae_normal:,.0f}")
print(f"R²   : {r2_normal:.4f}")""")
)

cells.append(md("""## 2.3 Régression avec statsmodels"""))

cells.append(
    code("""import statsmodels.api as sm

# Ajouter la constante
X_sm = sm.add_constant(X_train_processed)
model_sm = sm.OLS(y_train_reg.values, X_sm).fit()

print(model_sm.summary())""")
)

cells.append(
    code("""# Prédictions statsmodels
X_sm_test = sm.add_constant(X_test_processed)
y_pred_sm = model_sm.predict(X_sm_test)

mse_sm = mean_squared_error(y_test_reg, y_pred_sm)
r2_sm = r2_score(y_test_reg, y_pred_sm)
print(f"\\nstatsmodels OLS : MSE: {mse_sm:,.0f} | R²: {r2_sm:.4f}")""")
)

cells.append(md("""## 2.4 sklearn : Comparaison de modèles de régression"""))

cells.append(
    code("""from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import time

# Dictionnaire des modèles
reg_models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'SVR (RBF)': SVR(kernel='rbf'),
}

reg_results = {}

for name, model in reg_models.items():
    start = time.time()
    model.fit(X_train_processed, y_train_reg)
    train_time = time.time() - start
    
    y_pred_train = model.predict(X_train_processed)
    y_pred_test = model.predict(X_test_processed)
    
    reg_results[name] = {
        'MSE_train': mean_squared_error(y_train_reg, y_pred_train),
        'MSE_test': mean_squared_error(y_test_reg, y_pred_test),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_reg, y_pred_test)),
        'MAE_test': mean_absolute_error(y_test_reg, y_pred_test),
        'R2_train': r2_score(y_train_reg, y_pred_train),
        'R2_test': r2_score(y_test_reg, y_pred_test),
        'Training_time': train_time
    }
    
    print(f"{name:25s} | R² train: {reg_results[name]['R2_train']:.4f} | "
          f"R² test: {reg_results[name]['R2_test']:.4f} | "
          f"RMSE: {reg_results[name]['RMSE_test']:,.0f} | "
          f"Time: {train_time:.3f}s")""")
)

cells.append(md("## 2.5 Validation croisée"))

cells.append(
    code("""print("Validation croisée (5-fold) sur l'ensemble d'entraînement :\\n")

cv_results = {}
for name, model_class in [
    ('LinearRegression', LinearRegression()),
    ('DecisionTree', DecisionTreeRegressor(random_state=42)),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
]:
    scores = cross_val_score(model_class, X_train_processed, y_train_reg,
                             cv=5, scoring='r2', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:25s} | R² moyen: {scores.mean():.4f} ± {scores.std():.4f} | "
          f"Scores: [{', '.join(f'{s:.4f}' for s in scores)}]")""")
)

cells.append(md("## 2.6 Visualisation des résultats de régression"))

cells.append(
    code("""# Tableau comparatif
reg_df = pd.DataFrame(reg_results).T
reg_df = reg_df.sort_values('R2_test', ascending=False)

print("\\n=== Tableau comparatif des modèles de régression ===\\n")
print(reg_df[['R2_train', 'R2_test', 'RMSE_test', 'MAE_test', 'Training_time']].to_string())

# Détection overfitting
reg_df['Overfit_gap'] = reg_df['R2_train'] - reg_df['R2_test']
print("\\n--- Détection d'overfitting ---")
for idx, row in reg_df.iterrows():
    status = "⚠️ OVERFITTING" if row['Overfit_gap'] > 0.1 else "✓ OK"
    print(f"{idx:25s} | Gap R²: {row['Overfit_gap']:.4f} | {status}")""")
)

cells.append(
    code("""# Graphique comparatif R²
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# R² comparaison train vs test
x = np.arange(len(reg_df))
width = 0.35
axes[0].bar(x - width/2, reg_df['R2_train'], width, label='Train', color='steelblue')
axes[0].bar(x + width/2, reg_df['R2_test'], width, label='Test', color='coral')
axes[0].set_xticks(x)
axes[0].set_xticklabels(reg_df.index, rotation=45, ha='right')
axes[0].set_ylabel('R²')
axes[0].set_title('R² Train vs Test')
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# RMSE comparaison
axes[1].barh(reg_df.index, reg_df['RMSE_test'], color='coral', edgecolor='black')
axes[1].set_xlabel('RMSE')
axes[1].set_title('RMSE sur le Test Set')

plt.suptitle('Comparaison des modèles de régression (cible : MonthlyIncome)', fontsize=14)
plt.tight_layout()
plt.show()""")
)

cells.append(
    code("""# Prédictions vs valeurs réelles (meilleur modèle)
best_reg_name = reg_df.index[0]
best_reg_model = reg_models[best_reg_name]
y_pred_best = best_reg_model.predict(X_test_processed)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_test_reg, y_pred_best, alpha=0.5, s=20, color='steelblue')
axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()],
             'r--', linewidth=2, label='Prédiction parfaite')
axes[0].set_xlabel('Valeurs réelles')
axes[0].set_ylabel('Prédictions')
axes[0].set_title(f'{best_reg_name} : Prédictions vs Réel')
axes[0].legend()

# Résidus
residuals = y_test_reg.values - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.5, s=20, color='coral')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Prédictions')
axes[1].set_ylabel('Résidus')
axes[1].set_title(f'{best_reg_name} : Résidus')

plt.tight_layout()
plt.show()""")
)

cells.append(
    md("""## 2.7 Synthèse Régression

**Observations** :
- La régression linéaire fournit un baseline solide pour prédire le MonthlyIncome
- Les modèles ensemblistes (RandomForest, GradientBoosting) offrent généralement de meilleures performances
- Le DecisionTree tend à overfitter (R² train ≈ 1.0 vs R² test plus bas)
- La validation croisée confirme la stabilité des modèles

---""")
)

# =============================================================================
# PART 3 : CLASSIFICATION
# =============================================================================
cells.append(
    md("""# Partie 3 : Classification

> Inspiré du Workshop « Classification »

**Objectif** : Prédire le départ des employés (`Attrition` : 0/1).  
On compare 8 classifieurs avec gestion du déséquilibre de classes.""")
)

cells.append(md("## 3.1 Préparation des données pour la classification"))

cells.append(
    code("""from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, average_precision_score)

# Variable cible pour la classification
target_clf = 'Attrition'

X_clf = df.drop(columns=[target_clf])
y_clf = df[target_clf].copy()

# Identifier colonnes
num_features_clf = X_clf.select_dtypes(include=[np.number]).columns.tolist()
cat_features_clf = X_clf.select_dtypes(include=['object']).columns.tolist()

# Preprocesseur
preprocessor_clf = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features_clf),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features_clf)
    ]
)

# Split stratifié
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Prétraiter
X_train_clf_proc = preprocessor_clf.fit_transform(X_train_clf)
X_test_clf_proc = preprocessor_clf.transform(X_test_clf)

print(f"Distribution cible (train) : {dict(pd.Series(y_train_clf).value_counts())}")
print(f"Distribution cible (test)  : {dict(pd.Series(y_test_clf).value_counts())}")
print(f"X_train: {X_train_clf_proc.shape} | X_test: {X_test_clf_proc.shape}")""")
)

cells.append(md("## 3.2 Entraînement des classifieurs"))

cells.append(
    code("""# Dictionnaire des classifieurs
classifiers = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42),
    'Perceptron': Perceptron(
        max_iter=1000, random_state=42),
    'SVM (RBF)': SVC(
        kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42),
}

clf_results = {}

for name, clf in classifiers.items():
    start = time.time()
    clf.fit(X_train_clf_proc, y_train_clf)
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test_clf_proc)
    
    # Probabilités (si disponibles)
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_clf_proc)[:, 1]
        auc = roc_auc_score(y_test_clf, y_proba)
    elif hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(X_test_clf_proc)
        auc = roc_auc_score(y_test_clf, y_scores)
    else:
        auc = np.nan
        y_proba = None
    
    clf_results[name] = {
        'Accuracy': accuracy_score(y_test_clf, y_pred),
        'Precision': precision_score(y_test_clf, y_pred, zero_division=0),
        'Recall': recall_score(y_test_clf, y_pred, zero_division=0),
        'F1': f1_score(y_test_clf, y_pred, zero_division=0),
        'AUC-ROC': auc,
        'Training_time': train_time,
        'y_pred': y_pred,
    }
    
    print(f"{name:25s} | Acc: {clf_results[name]['Accuracy']:.4f} | "
          f"Prec: {clf_results[name]['Precision']:.4f} | "
          f"Rec: {clf_results[name]['Recall']:.4f} | "
          f"F1: {clf_results[name]['F1']:.4f} | "
          f"AUC: {clf_results[name]['AUC-ROC']:.4f}" if not np.isnan(auc) else
          f"{name:25s} | Acc: {clf_results[name]['Accuracy']:.4f} | "
          f"Prec: {clf_results[name]['Precision']:.4f} | "
          f"Rec: {clf_results[name]['Recall']:.4f} | "
          f"F1: {clf_results[name]['F1']:.4f} | "
          f"AUC: N/A")""")
)

cells.append(md("## 3.3 Rapports de classification détaillés"))

cells.append(
    code("""for name in classifiers:
    print(f"\\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(classification_report(y_test_clf, clf_results[name]['y_pred'],
                                target_names=['No (0)', 'Yes (1)']))""")
)

# =============================================================================
# PART 4 : COMPARATIVE ANALYSIS
# =============================================================================
cells.append(md("""# Partie 4 : Analyse Comparative et Recommandations"""))

cells.append(md("## 4.1 Tableau comparatif global"))

cells.append(
    code("""# Créer le DataFrame de résultats (sans y_pred)
results_display = {name: {k: v for k, v in m.items() if k != 'y_pred'}
                   for name, m in clf_results.items()}
clf_df = pd.DataFrame(results_display).T
clf_df = clf_df.sort_values('F1', ascending=False)

# Affichage formaté
print("=== Tableau comparatif des classifieurs ===\\n")
display_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'Training_time']
print(clf_df[display_cols].to_string(float_format=lambda x: f'{x:.4f}'))

print(f"\\n🏆 Meilleur F1-Score : {clf_df.index[0]} ({clf_df['F1'].iloc[0]:.4f})")""")
)

cells.append(md("## 4.2 Matrices de confusion"))

cells.append(
    code("""n_models = len(classifiers)
n_cols_cm = 4
n_rows_cm = (n_models + n_cols_cm - 1) // n_cols_cm

fig, axes = plt.subplots(n_rows_cm, n_cols_cm, figsize=(20, 5 * n_rows_cm))
axes = axes.flatten()

for i, (name, clf) in enumerate(classifiers.items()):
    cm = confusion_matrix(y_test_clf, clf_results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(name, fontsize=11)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Matrices de confusion : Tous les classifieurs', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 4.3 Courbes ROC"))

cells.append(
    code("""plt.figure(figsize=(10, 8))

for name, clf in classifiers.items():
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_clf_proc)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_proba = clf.decision_function(X_test_clf_proc)
    else:
        continue
    
    fpr, tpr, _ = roc_curve(y_test_clf, y_proba)
    auc_val = roc_auc_score(y_test_clf, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')
plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
plt.title('Courbes ROC : Comparaison des classifieurs', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 4.4 Courbes Precision-Recall"))

cells.append(
    code("""plt.figure(figsize=(10, 8))

for name, clf in classifiers.items():
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_clf_proc)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_proba = clf.decision_function(X_test_clf_proc)
    else:
        continue
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_clf, y_proba)
    ap = average_precision_score(y_test_clf, y_proba)
    plt.plot(recall_vals, precision_vals, label=f'{name} (AP={ap:.3f})', linewidth=2)

baseline = y_test_clf.mean()
plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Courbes Precision-Recall : Comparaison des classifieurs', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 4.5 Importance des features"))

cells.append(
    code("""# Feature importance pour RandomForest et GradientBoosting
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Récupérer les noms de features après preprocessing
feature_names = (num_features_clf +
                 list(preprocessor_clf.named_transformers_['cat']
                      .named_steps['encoder']
                      .get_feature_names_out(cat_features_clf)))

for ax, (name, model_key) in zip(axes, [('Random Forest', 'Random Forest'),
                                         ('Gradient Boosting', 'Gradient Boosting')]):
    clf_model = classifiers[model_key]
    importances = clf_model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15
    
    ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'{name} : Top 15 Features', fontsize=13)

plt.suptitle('Importance des features pour la prédiction d\\'Attrition', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 4.6 Visualisation comparative finale"))

cells.append(
    code("""# Bar chart comparatif de toutes les métriques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for ax, metric, color in zip(axes.flatten(), metrics_to_plot, colors_metrics):
    values = clf_df[metric].sort_values(ascending=True)
    values.plot(kind='barh', ax=ax, color=color, edgecolor='black')
    ax.set_title(metric, fontsize=14)
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.suptitle('Comparaison des classifieurs : Toutes les métriques', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(
    md("""## 4.7 Recommandations et Conclusion

### Choix du modèle
Dans le contexte RH de HumanForYou, le **Recall** est la métrique la plus importante :
- Un **faux négatif** (employé à risque non détecté) a un coût élevé → perte de talent, coût de remplacement (6-9 mois de salaire)
- Un **faux positif** (employé stable signalé à tort) a un coût faible → attention RH supplémentaire, entretien de suivi

→ Privilégier le modèle avec le **meilleur Recall** tout en conservant un F1-Score acceptable.

### Leviers d'action RH identifiés

Les variables retenues dans le modèle pointent vers des **leviers organisationnels**, non des caractéristiques individuelles :

1. **Satisfaction au travail** (EnvironmentSatisfaction, JobSatisfaction) → Enquêtes régulières, amélioration des conditions de travail, espaces collaboratifs
2. **Équilibre vie pro/perso** (WorkLifeBalance) → Flexibilité horaire, télétravail, droit à la déconnexion
3. **Évolution de carrière** (YearsSinceLastPromotion, YearsAtCompany) → Plans de carrière individualisés, revues annuelles, mobilité interne
4. **Rémunération** (MonthlyIncome, PercentSalaryHike, StockOptionLevel) → Benchmarks salariaux sectoriels, révisions ciblées, intéressement
5. **Engagement** (JobInvolvement, TrainingTimesLastYear) → Budget formation, responsabilisation, mentorat

### Bilan éthique

#### Ce qui a été fait
- **Retrait des critères protégés** : Gender, Age, MaritalStatus exclus du modèle pour éviter toute discrimination directe (Code du travail L.1132-1)
- **Retrait des données de surveillance** : avg_work_hours, std_work_hours, days_absent supprimés pour éviter le profilage comportemental (RGPD art. 22) et la pénalisation des arrêts maladie/congés parentaux
- **Focus organisationnel** : le modèle identifie des tendances structurelles, pas des profils individuels à risque

#### Risques résiduels à surveiller
- **EducationField** pourrait être un proxy du genre (filières genrées) → surveiller les prédictions par sous-groupe via audit de disparate impact
- **MonthlyIncome** reflète des inégalités salariales historiques → utiliser pour corriger les écarts, pas pour les justifier
- **DistanceFromHome** a été retirée car non actionnable par l'entreprise (on ne peut pas déplacer les employés)

#### Cadre d'usage recommandé
1. **Pas de décision individuelle automatisée** : le modèle est un outil d'aide à la décision RH globale (art. 22 RGPD)
2. **Transparence** : communiquer aux représentants du personnel l'existence et l'objectif du modèle
3. **Audit régulier** : vérifier l'absence de biais indirect (disparate impact) sur les groupes protégés
4. **Droit d'accès** : tout employé doit pouvoir connaître les données utilisées (RGPD art. 15)
5. **Finalité limitée** : les résultats ne doivent servir qu'à améliorer les conditions de travail, jamais à évaluer ou sanctionner

### Limites
- Dataset de taille modeste (~4700 employés d'une seule entreprise indienne)
- Données transversales (pas de suivi longitudinal : on ne sait pas *quand* les facteurs ont changé)
- Variables auto-déclarées (enquêtes de satisfaction → biais de désirabilité sociale)
- Le retrait des variables protégées réduit la capacité prédictive mais c'est un compromis éthique assumé
- Pas d'audit de disparate impact possible sans les variables protégées en colonne de test (un audit séparé est nécessaire)""")
)

# =============================================================================
# ASSEMBLE NOTEBOOK
# =============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("src/main.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {len(cells)} cells")
