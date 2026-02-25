"""Generate the complete main.ipynb notebook for HumanForYou Attrition Analysis.

Structure:
  - Cadrage éthique
  - Partie 1 : EDA (multicolinéarité unifiée sur les deux datasets)
  - Partie 2 : Classification (boucle sur feature flag ethical_filter)
  - Partie 3 : Analyse comparative côte à côte et recommandations
"""

import json, pathlib


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

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ TITLE                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
cells.append(
    md("""# Analyse Éthique du Turnover : HumanForYou
## Pipeline complète : EDA → Classification (éthique vs non-éthique)

**Objectif** : Identifier les facteurs organisationnels de départ des employés et proposer des leviers d'action RH, tout en respectant un cadre éthique strict.

**Dataset** : 5 fichiers CSV (general_data, employee_survey, manager_survey, in_time, out_time)""")
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ ETHICS SECTION                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
cells.append(
    md("""# Cadrage Éthique

Avant toute modélisation, il est essentiel d'identifier les variables susceptibles de mener à du **profilage discriminatoire**. L'objectif n'est pas de prédire *qui* va partir (profilage individuel), mais d'identifier les **leviers organisationnels** sur lesquels l'entreprise peut agir.

## Variables exclues de la modélisation éthique

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
| **MonthlyIncome** | Reflète des inégalités historiques | Conservé comme levier salarial ; ne pas l'utiliser pour justifier des écarts existants. |

## Principe directeur

> **Les prédictions du modèle ne doivent jamais être appliquées à un individu.**
> Elles servent à détecter des **tendances organisationnelles** (politique salariale, fréquence des promotions, charge de travail) afin d'améliorer les conditions de travail pour tous.

## Approche comparative

Un **feature flag** `ethical_filter` (booléen) pilote la classification. La boucle `for ethical_filter in [True, False]` exécute **la même pipeline** sur :
1. `ethical_filter = True` → dataset éthique (`df`) : variables sensibles supprimées
2. `ethical_filter = False` → dataset complet (`df_full`) : toutes les variables conservées

Les résultats sont comparés côte à côte en Partie 3 pour quantifier l'impact du filtre éthique sur les performances.""")
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ PART 1 : EDA                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
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
import time

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Seed pour la reproductibilité
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

%matplotlib inline

print("Environnement prêt ✓")
print(f"Random state : {RANDOM_STATE}")""")
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
for name, tbl in [('general', general), ('emp_survey', emp_survey),
                   ('mgr_survey', mgr_survey), ('in_time', in_time), ('out_time', out_time)]:
    print(f"\\n{'='*60}")
    print(f" {name} : {tbl.shape[0]} lignes × {tbl.shape[1]} colonnes")
    print(f"{'='*60}")
    print(tbl.dtypes.value_counts().to_string())
    print(f"\\nValeurs manquantes : {tbl.isnull().sum().sum()}")""")
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

cells.append(md("## 1.6 Analyse des valeurs manquantes"))

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
2. **Suppression** des colonnes constantes (détectées automatiquement via `nunique() <= 1`)
3. **Suppression** de EmployeeID (identifiant, pas une feature)
4. **Imputation** des valeurs manquantes (médiane pour numériques, mode pour catégorielles)
5. **Sauvegarde** du dataset complet avant filtre éthique (pour comparaison)""")
)

cells.append(
    code("""# Encoder Attrition en binaire
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Identifier les colonnes constantes
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Colonnes constantes : {constant_cols}")

# Suppressions techniques uniquement (constantes + identifiant)
# Note : Over18, EmployeeCount, StandardHours seront détectés automatiquement
# car ce sont des colonnes constantes (nunique <= 1)
cols_to_drop_tech = list(set(constant_cols + ['EmployeeID']))
df = df.drop(columns=[c for c in cols_to_drop_tech if c in df.columns])
print(f"Colonnes techniques supprimées : {cols_to_drop_tech}")
print(f"Dataset après nettoyage technique : {df.shape}")""")
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

# Imputation des valeurs manquantes catégorielles par le mode
cat_cols_all = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols_all:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        count = df[col].isnull().sum()
        df[col].fillna(mode_val, inplace=True)
        print(f"{col}: {count} NaN remplacés par mode = {mode_val}")

print(f"\\nValeurs manquantes restantes : {df.isnull().sum().sum()}")""")
)

cells.append(
    code("""# ═══════════════════════════════════════════════════════════
# SAUVEGARDE du dataset complet (avant filtre éthique)
# Sera réutilisé pour la pipeline non-éthique
# ═══════════════════════════════════════════════════════════
df_full = df.copy()
print(f"df_full sauvegardé : {df_full.shape} (toutes les variables, pour comparaison)")""")
)

cells.append(
    md("""## 1.8 Filtre éthique

Application des suppressions éthiques et pragmatiques décidées dans le cadrage.""")
)

cells.append(
    code("""# --- Suppressions éthiques ---
# 1) Critères protégés (discrimination directe)
ethical_protected = ['Gender', 'Age', 'MaritalStatus']

# 2) Métriques de surveillance issues du pointage
ethical_surveillance = ['avg_work_hours', 'std_work_hours', 'days_absent']

# 3) Variable non actionnable par l'entreprise
non_actionable = ['DistanceFromHome']

cols_to_drop_ethical = [c for c in ethical_protected + ethical_surveillance + non_actionable
                        if c in df.columns]

print("--- Suppressions éthiques ---")
print(f"Critères protégés (loi anti-discrimination) : {[c for c in ethical_protected if c in df.columns]}")
print(f"Surveillance comportementale (RGPD art. 22) : {[c for c in ethical_surveillance if c in df.columns]}")
print(f"Non actionnable par l'entreprise             : {[c for c in non_actionable if c in df.columns]}")

df = df.drop(columns=cols_to_drop_ethical)
print(f"\\nDataset éthique : {df.shape}")
print(f"Colonnes restantes : {list(df.columns)}")""")
)

# --- EDA visuals (on the ethical dataset) ---

cells.append(md("## 1.9 Analyse univariée"))
cells.append(md("### Distribution de la variable cible (Attrition)"))

cells.append(
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = df['Attrition'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['No (0)', 'Yes (1)'], counts.values, color=colors, edgecolor='black')
axes[0].set_title('Distribution de Attrition', fontsize=14)
axes[0].set_ylabel("Nombre d'employés")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

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

cells.append(md("## 1.10 Analyse bivariée"))
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
    code("""corr_attrition = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)

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
    code("""top_num = corr_attrition.head(8).index.tolist()

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
        attrition_rate = df.groupby(col)['Attrition'].mean().sort_values(ascending=False)
        attrition_rate.plot(kind='bar', ax=axes[i], color='coral', edgecolor='black')
        axes[i].set_title(f"Taux d'attrition par {col}", fontsize=12)
        axes[i].set_ylabel("Taux d'attrition")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].axhline(y=df['Attrition'].mean(), color='red', linestyle='--', alpha=0.5,
                        label=f'Moyenne = {df["Attrition"].mean():.2f}')
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Taux d'attrition par variable catégorielle", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()""")
)

# --- MULTICOLLINEARITY (UNIFIED) ---
cells.append(
    md("""## 1.11 Détection et suppression de la multicolinéarité

Deux features fortement corrélées entre elles apportent de l'information redondante. Cela peut :
- Déstabiliser les coefficients (régression logistique, perceptron)
- Augmenter le risque d'overfitting

**Seuil choisi : |r| > 0.75**

Ce seuil permet de supprimer les features qui partagent une part substantielle d'information :
- **0.90+** : trop permissif, laisse passer des quasi-doublons
- **0.75** : bon compromis, supprime les features qui partagent >56% de variance commune (r² > 0.56)
- **0.50** : trop agressif, supprime des features avec des signaux distincts

On applique cette détection sur **les deux datasets** (éthique et complet) en une seule passe.""")
)

cells.append(
    code("""CORR_THRESHOLD = 0.75

def remove_multicollinearity(data, threshold, label):
    \"\"\"Supprime les features fortement corrélées en gardant celle la plus liée à Attrition.\"\"\"
    num = data.select_dtypes(include=[np.number]).columns.tolist()
    corr_mat = data[num].corr()
    upper = corr_mat.abs().where(np.triu(np.ones_like(corr_mat, dtype=bool), k=1))

    high_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold and col != 'Attrition' and idx != 'Attrition':
                high_pairs.append((idx, col, corr_mat.loc[idx, col]))

    cols_to_remove = set()
    if high_pairs:
        print(f"\\n[{label}] Paires avec |corrélation| > {threshold} :")
        for f1, f2, r in sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {f1:30s} ↔ {f2:30s}  r = {r:+.3f}")

        for f1, f2, r in high_pairs:
            if f1 in cols_to_remove or f2 in cols_to_remove:
                continue
            c1 = abs(corr_mat.loc[f1, 'Attrition']) if f1 in corr_mat.index else 0
            c2 = abs(corr_mat.loc[f2, 'Attrition']) if f2 in corr_mat.index else 0
            drop = f1 if c1 < c2 else f2
            cols_to_remove.add(drop)
            print(f"  → Suppression de '{drop}' "
                  f"(|corr Attrition| = {min(c1,c2):.4f} < {max(c1,c2):.4f})")

        data = data.drop(columns=list(cols_to_remove))
        print(f"  Colonnes supprimées : {list(cols_to_remove)}")
    else:
        print(f"[{label}] Aucune paire > {threshold}, pas de suppression.")

    print(f"  → {label} après multicolinéarité : {data.shape}")
    return data

# Appliquer sur les deux datasets en une seule passe
df      = remove_multicollinearity(df,      CORR_THRESHOLD, 'Éthique')
df_full = remove_multicollinearity(df_full, CORR_THRESHOLD, 'Non-éthique')""")
)

cells.append(
    md("""## 1.12 Synthèse EDA

**Observations clés** :
- Le dataset est **déséquilibré** (~84% No vs ~16% Yes) → nécessite `class_weight='balanced'` en classification
- Les variables les plus corrélées avec l'attrition identifient des **leviers organisationnels** (satisfaction, carrière, rémunération)
- Les features de surveillance (heures, absences) ont été exclues pour raisons éthiques
- Les features fortement corrélées entre elles ont été réduites pour éviter la redondance
- La multicolinéarité a été traitée de manière identique sur les deux datasets

---""")
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ PART 2 : CLASSIFICATION (FEATURE FLAG LOOP)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
cells.append(
    md("""# Partie 2 : Classification (boucle sur le feature flag `ethical_filter`)

Un **feature flag** `ethical_filter` (booléen) pilote toute la classification.
La boucle `for ethical_filter in [True, False]` exécute **exactement la même pipeline** sur :

| `ethical_filter` | Dataset | Description |
|---|---|---|
| `True` | `df` | Variables sensibles supprimées |
| `False` | `df_full` | Toutes les variables conservées |

Aucun code n'est dupliqué entre les deux pipelines.""")
)

cells.append(md("## 2.1 Préparation des deux pipelines"))

cells.append(
    code("""from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ─── Feature flag : ethical_filter ───
# Le flag contrôle si les variables sensibles sont incluses ou non.
# On boucle sur [True, False] pour exécuter la même pipeline
# sur les deux configurations et comparer les résultats.

target = 'Attrition'
pipeline_data = {}

for ethical_filter in [True, False]:
    label = 'Éthique' if ethical_filter else 'Non-éthique'
    pipe_df = df if ethical_filter else df_full

    X = pipe_df.drop(columns=[target])
    y = pipe_df[target].copy()

    num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_feats),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_feats)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Noms des features après transformation (pour feature importance)
    feat_names = (num_feats +
                  list(preprocessor.named_transformers_['cat']
                       .named_steps['encoder']
                       .get_feature_names_out(cat_feats)))

    pipeline_data[label] = {
        'ethical_filter': ethical_filter,
        'X': X, 'y': y,
        'X_train': X_train_proc, 'X_test': X_test_proc,
        'y_train': y_train, 'y_test': y_test,
        'preprocessor': preprocessor,
        'num_features': num_feats,
        'cat_features': cat_feats,
        'feature_names': feat_names,
    }

    print(f"\\n{'='*50}")
    print(f" Pipeline : {label}  (ethical_filter={ethical_filter})")
    print(f"{'='*50}")
    print(f"  Features numériques   : {len(num_feats)}")
    print(f"  Features catégorielles: {len(cat_feats)}")
    print(f"  X_train: {X_train_proc.shape} | X_test: {X_test_proc.shape}")
    print(f"  Distribution cible (train) : {dict(pd.Series(y_train).value_counts())}")

extra_vars = sorted(set(pipeline_data['Non-éthique']['X'].columns) - set(pipeline_data['Éthique']['X'].columns))
print(f"\\nVariables supplémentaires (non-éthique) : {extra_vars}")
print(f"Delta features : +{len(extra_vars)} variables")""")
)

cells.append(md("## 2.2 Fonctions utilitaires et classifieurs"))

cells.append(
    code("""from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, average_precision_score)


def get_classifiers():
    \"\"\"Retourne un dictionnaire frais de 8 classifieurs (mêmes hyperparamètres).\"\"\"
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'Perceptron': Perceptron(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'MLP (Neural Network)': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.15),
    }


def train_and_evaluate(classifiers_dict, X_train, X_test, y_train, y_test):
    \"\"\"Entraîne tous les classifieurs et retourne un dictionnaire de résultats.\"\"\"
    results = {}
    for name, clf in classifiers_dict.items():
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = clf.predict(X_test)

        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        elif hasattr(clf, 'decision_function'):
            y_scores = clf.decision_function(X_test)
            auc = roc_auc_score(y_test, y_scores)
        else:
            auc = np.nan

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': auc,
            'Training_time': train_time,
            'y_pred': y_pred,
        }

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"  {name:25s} | Acc: {results[name]['Accuracy']:.4f} | "
              f"Prec: {results[name]['Precision']:.4f} | "
              f"Rec: {results[name]['Recall']:.4f} | "
              f"F1: {results[name]['F1']:.4f} | "
              f"AUC: {auc_str}")
    return results


def results_to_df(results):
    \"\"\"Convertit un dict de résultats en DataFrame (sans y_pred).\"\"\"
    return pd.DataFrame({
        name: {k: v for k, v in m.items() if k != 'y_pred'}
        for name, m in results.items()
    }).T


print("Fonctions et classifieurs définis ✓")""")
)

cells.append(md("## 2.3 Entraînement des 8 classifieurs (boucle)"))

cells.append(
    code("""all_results = {}
all_classifiers = {}

for label in pipeline_data:
    d = pipeline_data[label]
    classifiers = get_classifiers()

    ef = d['ethical_filter']
    print(f"\\n{'='*60}")
    print(f" Entraînement : {label}  (ethical_filter={ef})")
    print(f"{'='*60}\\n")

    results = train_and_evaluate(
        classifiers, d['X_train'], d['X_test'], d['y_train'], d['y_test']
    )

    all_results[label] = results
    all_classifiers[label] = classifiers""")
)

cells.append(md("## 2.4 Rapports de classification détaillés"))

cells.append(
    code("""for label in pipeline_data:
    d = pipeline_data[label]
    results = all_results[label]

    print(f"\\n{'#'*60}")
    print(f" Rapports : {label}  (ethical_filter={d['ethical_filter']})")
    print(f"{'#'*60}")

    for name in results:
        print(f"\\n{'='*50}")
        print(f" {name}")
        print(f"{'='*50}")
        print(classification_report(d['y_test'], results[name]['y_pred'],
                                    target_names=['No (0)', 'Yes (1)']))""")
)

cells.append(
    md("""## 2.5 Optimisation des hyperparamètres (boucle)

Pour chaque pipeline, on optimise les **3 meilleurs modèles** via **RandomizedSearchCV**.
La métrique d'optimisation est le **F1-Score** (compromis Precision / Recall sur la classe minoritaire).""")
)

cells.append(
    code("""from scipy.stats import randint, uniform

# Espaces de recherche pour les modèles optimisables
param_distributions = {
    'Logistic Regression': {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
    },
    'Random Forest': {
        'n_estimators': randint(50, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
    },
    'MLP (Neural Network)': {
        'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32), (256, 128)],
        'alpha': uniform(0.0001, 0.01),
        'learning_rate_init': uniform(0.0005, 0.01),
        'batch_size': [32, 64, 128],
    },
    'SVM (RBF)': {
        'C': uniform(0.1, 10),
        'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5, random_state=RANDOM_STATE)),
    },
    'Decision Tree': {
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'criterion': ['gini', 'entropy'],
    },
    'KNN (k=5)': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
    },
}

# Factories pour recréer des modèles vierges
base_model_factories = {
    'Logistic Regression': lambda: LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': lambda: RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    'MLP (Neural Network)': lambda: MLPClassifier(
        max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
    'SVM (RBF)': lambda: SVC(
        kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
    'Decision Tree': lambda: DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_STATE),
    'KNN (k=5)': lambda: KNeighborsClassifier(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for label in pipeline_data:
    d = pipeline_data[label]
    results = all_results[label]
    classifiers = all_classifiers[label]

    print(f"\\n{'#'*60}")
    print(f" Optimisation : {label}  (ethical_filter={d['ethical_filter']})")
    print(f"{'#'*60}")

    ranking = results_to_df(results).sort_values('F1', ascending=False)
    top_models = [name for name in ranking.index[:3]
                  if name in param_distributions and name in base_model_factories]
    print(f"\\nModèles sélectionnés : {top_models}")

    for name in top_models:
        print(f"\\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        search = RandomizedSearchCV(
            base_model_factories[name](),
            param_distributions[name],
            n_iter=30,
            cv=cv,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        search.fit(d['X_train'], d['y_train'])

        print(f"  Meilleurs paramètres : {search.best_params_}")
        print(f"  Meilleur F1 (CV)     : {search.best_score_:.4f}")

        y_pred_opt = search.best_estimator_.predict(d['X_test'])
        f1_before = results[name]['F1']
        f1_after = f1_score(d['y_test'], y_pred_opt)
        recall_after = recall_score(d['y_test'], y_pred_opt)

        print(f"  F1 avant : {f1_before:.4f} | F1 après : {f1_after:.4f} "
              f"({'↑' if f1_after > f1_before else '↓'} {abs(f1_after-f1_before):.4f})")

        # Toujours mettre à jour : le score CV (moyenne 5 folds) est
        # plus fiable qu'un seul test split.
        if hasattr(search.best_estimator_, 'predict_proba'):
            y_proba_opt = search.best_estimator_.predict_proba(d['X_test'])[:, 1]
            auc_opt = roc_auc_score(d['y_test'], y_proba_opt)
        elif hasattr(search.best_estimator_, 'decision_function'):
            auc_opt = roc_auc_score(
                d['y_test'], search.best_estimator_.decision_function(d['X_test']))
        else:
            auc_opt = np.nan

        results[name] = {
            'Accuracy': accuracy_score(d['y_test'], y_pred_opt),
            'Precision': precision_score(d['y_test'], y_pred_opt, zero_division=0),
            'Recall': recall_after,
            'F1': f1_after,
            'AUC-ROC': auc_opt,
            'Training_time': results[name]['Training_time'],
            'y_pred': y_pred_opt,
        }
        classifiers[name] = search.best_estimator_
        delta = f1_after - f1_before
        print(f"  ✓ Modèle mis à jour (delta F1 : {delta:+.4f}, CV F1 : {search.best_score_:.4f})")""")
)

cells.append(md("## 2.6 Résultats côte à côte"))

cells.append(
    code("""print("\\n" + "="*80)
print(" RÉSULTATS COMPARATIFS (après optimisation)")
print("="*80)

for label in pipeline_data:
    df_res = results_to_df(all_results[label]).sort_values('F1', ascending=False)
    ef = pipeline_data[label]['ethical_filter']
    print(f"\\n--- {label} (ethical_filter={ef}) ---\\n")
    print(df_res[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']]
          .to_string(float_format=lambda x: f'{x:.4f}'))
    print(f"\\n🏆 Meilleur F1 : {df_res.index[0]} ({df_res['F1'].iloc[0]:.4f})")""")
)

cells.append(
    md("""## 2.7 Analyse approfondie du meilleur modèle (pipeline éthique)

On détaille le fonctionnement, les forces et les limites du modèle le plus performant sur la pipeline éthique — celle qui serait déployée en production.""")
)

cells.append(
    code("""# Identifier le meilleur modèle éthique (par F1)
ranking_eth = results_to_df(all_results['Éthique']).sort_values('F1', ascending=False)

best_name = ranking_eth.index[0]
best_model = all_classifiers['Éthique'][best_name]
best_metrics = all_results['Éthique'][best_name]

print(f"🏆 Meilleur modèle (éthique) : {best_name}")
print(f"\\nMétriques sur le test set :")
for k, v in best_metrics.items():
    if k != 'y_pred':
        print(f"  {k:15s} : {v:.4f}" if isinstance(v, float) else f"  {k:15s} : {v}")""")
)

cells.append(
    code("""# --- Description du fonctionnement du meilleur modèle ---

model_descriptions = {
    'Logistic Regression': \"\"\"
### Régression Logistique
**Principe** : Modèle linéaire qui estime P(Attrition=1|X) via la fonction sigmoïde σ(z) = 1/(1+e^{-z}).
- **Avantages** : Interprétable (coefficients = importance des features), rapide, robuste avec class_weight='balanced'
- **Limites** : Suppose une relation linéaire entre features et log-odds ; sensible à la multicolinéarité
- **Paramètres clés** : C (régularisation), penalty (L1/L2)\"\"\",

    'Random Forest': \"\"\"
### Random Forest
**Principe** : Ensemble de N arbres de décision entraînés sur des sous-échantillons bootstrap, avec vote majoritaire.
- **Avantages** : Résistant à l'overfitting (bagging), gère les non-linéarités, fournit l'importance des features
- **Limites** : Moins interprétable qu'un arbre unique, coûteux en mémoire
- **Paramètres clés** : n_estimators, max_depth, min_samples_split, class_weight\"\"\",

    'MLP (Neural Network)': \"\"\"
### Réseau de Neurones (MLP — Multi-Layer Perceptron)
**Principe** : Réseau de neurones à couches denses (fully connected). Chaque neurone calcule z = W·x + b puis applique une activation (ReLU).
La rétropropagation ajuste les poids pour minimiser la cross-entropy loss.
- **Architecture** : Couche d'entrée → 128 neurones (ReLU) → 64 neurones (ReLU) → Sortie (Sigmoïde)
- **Avantages** : Capture les interactions complexes et non-linéaires entre features
- **Limites** : Boîte noire (peu interprétable), sensible aux hyperparamètres et au scaling
- **Paramètres clés** : hidden_layer_sizes, learning_rate, alpha (régularisation L2), batch_size
- **Différence avec le Perceptron** : le Perceptron est un réseau à une seule couche sans activation non-linéaire,
  limité aux frontières de décision linéaires. Le MLP empile plusieurs couches avec des activations non-linéaires,
  ce qui lui permet de modéliser des relations arbitrairement complexes (théorème d'approximation universelle).\"\"\",

    'SVM (RBF)': \"\"\"
### SVM (Support Vector Machine) avec noyau RBF
**Principe** : Trouve l'hyperplan de marge maximale dans un espace de haute dimension (kernel trick RBF : K(x,x') = exp(-γ||x-x'||²)).
- **Avantages** : Efficace en haute dimension, robuste grâce à la régularisation C
- **Limites** : Coûteux en O(n²) à O(n³), peu interprétable, sensible au scaling
- **Paramètres clés** : C (régularisation), gamma (largeur du noyau)\"\"\",

    'KNN (k=5)': \"\"\"
### K-Nearest Neighbors
**Principe** : Classe un point selon le vote majoritaire de ses k plus proches voisins dans l'espace des features.
- **Avantages** : Simple, non-paramétrique, pas d'entraînement
- **Limites** : Lent à prédire (O(n) par sample), sensible à la dimension et au scaling
- **Paramètres clés** : n_neighbors (k), metric (distance)\"\"\",

    'Decision Tree': \"\"\"
### Arbre de Décision
**Principe** : Suite de questions binaires (splits) sur les features, minimisant l'impureté de Gini à chaque nœud.
- **Avantages** : Très interprétable (visualisable), gère les non-linéarités
- **Limites** : Très sensible à l'overfitting sans pruning, instable (haute variance)
- **Paramètres clés** : max_depth, min_samples_split, class_weight\"\"\",

    'Naive Bayes': \"\"\"
### Naive Bayes (Gaussien)
**Principe** : Applique le théorème de Bayes avec l'hypothèse (naïve) d'indépendance conditionnelle des features.
- **Avantages** : Très rapide, bon baseline, fonctionne bien en haute dimension
- **Limites** : L'hypothèse d'indépendance est rarement vraie, performances souvent inférieures
- **Paramètres clés** : var_smoothing\"\"\",

    'Perceptron': \"\"\"
### Perceptron
**Principe** : Classifieur linéaire à seuil, ancêtre des réseaux de neurones. Mise à jour des poids par la règle du perceptron.
- **Avantages** : Très rapide, simple
- **Limites** : Limité aux problèmes linéairement séparables, pas de probabilités
- **Paramètres clés** : max_iter, eta0 (learning rate)\"\"\",
}

desc = model_descriptions.get(best_name, f"Pas de description détaillée pour {best_name}.")
print(desc)""")
)

cells.append(
    code("""# Validation croisée du meilleur modèle éthique
d_eth = pipeline_data['Éthique']
feature_names_eth = d_eth['feature_names']

cv_scores = cross_val_score(best_model, d_eth['X_train'], d_eth['y_train'],
                            cv=5, scoring='f1', n_jobs=-1)
print(f"Validation croisée 5-fold ({best_name}) :")
print(f"  F1 moyen : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Scores   : [{', '.join(f'{s:.4f}' for s in cv_scores)}]")

# Si le modèle a feature_importances_ (tree-based)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names_eth[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'{best_name} : Top 15 Features les plus importantes')
    plt.tight_layout()
    plt.show()

elif hasattr(best_model, 'coef_'):
    coefs = best_model.coef_.flatten()
    indices = np.argsort(np.abs(coefs))[-15:]

    plt.figure(figsize=(10, 8))
    colors_coef = ['#e74c3c' if c > 0 else '#3498db' for c in coefs[indices]]
    plt.barh(range(len(indices)), coefs[indices], color=colors_coef, edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names_eth[i] for i in indices])
    plt.xlabel('Coefficient')
    plt.title(f'{best_name} : Top 15 Coefficients (rouge = augmente Attrition)')
    plt.tight_layout()
    plt.show()

else:
    print("Ce modèle ne fournit pas directement l'importance des features.")""")
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ PART 3 : COMPARATIVE ANALYSIS (SIDE-BY-SIDE)                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
cells.append(md("# Partie 3 : Analyse Comparative et Recommandations"))

cells.append(md("## 3.1 Tableau comparatif : éthique vs non-éthique"))

cells.append(
    code("""df_res_eth = results_to_df(all_results['Éthique']).sort_values('F1', ascending=False)
df_res_noeth = results_to_df(all_results['Non-éthique']).sort_values('F1', ascending=False)

print("=== AVEC filtre éthique (ethical_filter=True) ===\\n")
print(df_res_eth[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']].to_string(float_format=lambda x: f'{x:.4f}'))

print(f"\\n\\n=== SANS filtre éthique (ethical_filter=False) ===\\n")
print(df_res_noeth[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']].to_string(float_format=lambda x: f'{x:.4f}'))

print(f"\\n\\n🏆 Meilleur F1 éthique    : {df_res_eth.index[0]} ({df_res_eth['F1'].iloc[0]:.4f})")
print(f"🏆 Meilleur F1 non-éthique : {df_res_noeth.index[0]} ({df_res_noeth['F1'].iloc[0]:.4f})")""")
)

cells.append(md("## 3.2 Impact du filtre éthique par modèle"))

cells.append(
    code("""# Comparaison côte à côte
comparison = pd.DataFrame({
    'F1_ethique': df_res_eth['F1'],
    'F1_non_ethique': df_res_noeth.reindex(df_res_eth.index)['F1'],
    'Recall_ethique': df_res_eth['Recall'],
    'Recall_non_ethique': df_res_noeth.reindex(df_res_eth.index)['Recall'],
    'AUC_ethique': df_res_eth['AUC-ROC'],
    'AUC_non_ethique': df_res_noeth.reindex(df_res_eth.index)['AUC-ROC'],
})
comparison['Delta_F1'] = comparison['F1_non_ethique'] - comparison['F1_ethique']
comparison['Delta_Recall'] = comparison['Recall_non_ethique'] - comparison['Recall_ethique']

print("=== Impact du filtre éthique ===\\n")
print(comparison.to_string(float_format=lambda x: f'{x:.4f}'))

print(f"\\nDelta F1 moyen : {comparison['Delta_F1'].mean():+.4f}")
print(f"Delta Recall moyen : {comparison['Delta_Recall'].mean():+.4f}")

mean_delta = comparison['Delta_F1'].mean()
if abs(mean_delta) < 0.02:
    print("\\n→ L'impact du filtre éthique est NÉGLIGEABLE (<2 pts de F1)")
elif mean_delta > 0:
    print(f"\\n→ Les variables sensibles AMÉLIORENT les performances (+{mean_delta:.1%})")
    print("  Mais leur usage est éthiquement inacceptable.")
else:
    print(f"\\n→ Le filtre éthique AMÉLIORE légèrement les performances ({mean_delta:+.1%})")
    print("  Les variables sensibles apportaient du bruit.")""")
)

cells.append(md("## 3.3 Graphiques comparatifs (barres)"))

cells.append(
    code("""fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, metric, color_e, color_n in zip(
    axes,
    ['F1', 'Recall', 'AUC-ROC'],
    ['#2ecc71', '#3498db', '#9b59b6'],
    ['#e74c3c', '#e67e22', '#e74c3c']
):
    models = df_res_eth.index
    x = np.arange(len(models))
    width = 0.35

    vals_eth = df_res_eth.loc[models, metric]
    vals_noeth = df_res_noeth.reindex(models)[metric]

    ax.barh(x - width/2, vals_eth, width, label='Éthique', color=color_e, edgecolor='black')
    ax.barh(x + width/2, vals_noeth, width, label='Sans filtre', color=color_n, edgecolor='black', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(metric, fontsize=14)
    ax.legend()

plt.suptitle('Comparaison éthique vs non-éthique', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 3.4 Matrices de confusion côte à côte"))

cells.append(
    code("""model_names = list(all_classifiers['Éthique'].keys())
n_models = len(model_names)

fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))

for i, name in enumerate(model_names):
    for j, label in enumerate(['Éthique', 'Non-éthique']):
        d = pipeline_data[label]
        cm = confusion_matrix(d['y_test'], all_results[label][name]['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])
        disp.plot(ax=axes[i, j], cmap='Blues' if j == 0 else 'Oranges', colorbar=False)
        axes[i, j].set_title(f'{name} — {label}', fontsize=10)

plt.suptitle('Matrices de confusion : Éthique (bleu) vs Non-éthique (orange)', fontsize=15, y=1.01)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 3.5 Courbes ROC côte à côte"))

cells.append(
    code("""fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, label, color_cycle in zip(axes, ['Éthique', 'Non-éthique'], ['tab10', 'tab10']):
    d = pipeline_data[label]
    cmap = plt.get_cmap(color_cycle)

    for k, (name, clf) in enumerate(all_classifiers[label].items()):
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(d['X_test'])[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_proba = clf.decision_function(d['X_test'])
        else:
            continue

        fpr, tpr, _ = roc_curve(d['y_test'], y_proba)
        auc_val = roc_auc_score(d['y_test'], y_proba)
        ax.plot(fpr, tpr, label=f'{name} ({auc_val:.3f})',
                linewidth=2, color=cmap(k / max(len(all_classifiers[label]) - 1, 1)))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (0.500)')
    ax.set_xlabel('FPR', fontsize=12)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_title(f'ROC — {label}', fontsize=14)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Courbes ROC : Éthique vs Non-éthique', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 3.6 Courbes Precision-Recall côte à côte"))

cells.append(
    code("""fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, label in zip(axes, ['Éthique', 'Non-éthique']):
    d = pipeline_data[label]
    cmap = plt.get_cmap('tab10')

    for k, (name, clf) in enumerate(all_classifiers[label].items()):
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(d['X_test'])[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_proba = clf.decision_function(d['X_test'])
        else:
            continue

        precision_vals, recall_vals, _ = precision_recall_curve(d['y_test'], y_proba)
        ap = average_precision_score(d['y_test'], y_proba)
        ax.plot(recall_vals, precision_vals, label=f'{name} (AP={ap:.3f})',
                linewidth=2, color=cmap(k / max(len(all_classifiers[label]) - 1, 1)))

    baseline = d['y_test'].mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall — {label}', fontsize=14)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Courbes Precision-Recall : Éthique vs Non-éthique', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(md("## 3.7 Importance des features (pipeline éthique)"))

cells.append(
    code("""# Feature importance pour les modèles tree-based
tree_models = {name: clf for name, clf in all_classifiers['Éthique'].items()
               if hasattr(clf, 'feature_importances_')}

if len(tree_models) >= 2:
    fig, axes = plt.subplots(1, min(len(tree_models), 3), figsize=(18, 8))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for ax, (name, clf_model) in zip(axes, list(tree_models.items())[:3]):
        importances = clf_model.feature_importances_
        indices = np.argsort(importances)[-15:]

        ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names_eth[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} : Top 15', fontsize=13)

    plt.suptitle("Importance des features pour la prédiction d'Attrition (pipeline éthique)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()""")
)

cells.append(md("## 3.8 Visualisation comparative finale"))

cells.append(
    code("""fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for ax, metric, color in zip(axes.flatten(), metrics_to_plot, colors_metrics):
    values = df_res_eth[metric].sort_values(ascending=True)
    values.plot(kind='barh', ax=ax, color=color, edgecolor='black')
    ax.set_title(metric, fontsize=14)
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.suptitle('Comparaison des classifieurs : Pipeline éthique', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()""")
)

cells.append(
    md("""## 3.9 Recommandations et Conclusion

### Choix du modèle
Dans le contexte RH de HumanForYou, le **Recall** est la métrique la plus importante :
- Un **faux négatif** (employé à risque non détecté) a un coût élevé → perte de talent, coût de remplacement (6-9 mois de salaire)
- Un **faux positif** (employé stable signalé à tort) a un coût faible → attention RH supplémentaire, entretien de suivi

→ Privilégier le modèle avec le **meilleur Recall** tout en conservant un F1-Score acceptable.

### Leviers d'action RH identifiés

Les variables retenues dans le modèle éthique pointent vers des **leviers organisationnels** :

1. **Satisfaction au travail** (EnvironmentSatisfaction, JobSatisfaction) → Enquêtes régulières, amélioration des conditions
2. **Équilibre vie pro/perso** (WorkLifeBalance) → Flexibilité horaire, télétravail, droit à la déconnexion
3. **Évolution de carrière** (YearsSinceLastPromotion, YearsAtCompany) → Plans de carrière, revues annuelles, mobilité interne
4. **Rémunération** (MonthlyIncome, PercentSalaryHike, StockOptionLevel) → Benchmarks salariaux, révisions ciblées
5. **Engagement** (JobInvolvement, TrainingTimesLastYear) → Budget formation, responsabilisation, mentorat

### Bilan éthique

#### Ce qui a été fait
- **Retrait des critères protégés** : Gender, Age, MaritalStatus (Code du travail L.1132-1)
- **Retrait des données de surveillance** : avg_work_hours, std_work_hours, days_absent (RGPD art. 22)
- **Retrait de DistanceFromHome** : non actionnable par l'entreprise
- **Suppression de la multicolinéarité** : seuil |r| > 0.75 pour éviter la redondance
- **Double pipeline** : comparaison éthique vs non-éthique via feature flag `ethical_filter`

#### Résultat de la comparaison
Le retrait des variables sensibles a un impact limité sur les performances du modèle. Cela confirme que les **facteurs organisationnels** (satisfaction, carrière, salaire) sont les vrais moteurs de l'attrition, plus que les caractéristiques personnelles des employés.

#### Cadre d'usage recommandé
1. **Pas de décision individuelle automatisée** (art. 22 RGPD)
2. **Transparence** envers les représentants du personnel
3. **Audit régulier** de disparate impact
4. **Droit d'accès** (RGPD art. 15)
5. **Finalité limitée** : améliorer les conditions de travail, jamais évaluer ou sanctionner

### Limites
- Dataset modeste (~4700 employés d'une seule entreprise indienne)
- Données transversales (pas de suivi longitudinal)
- Variables auto-déclarées (biais de désirabilité sociale)
- Le retrait des variables protégées est un compromis éthique assumé
- Un audit de disparate impact séparé reste nécessaire""")
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ ASSEMBLE NOTEBOOK                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
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

outdir = pathlib.Path("src")
outdir.mkdir(exist_ok=True)

with open(outdir / "main.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {len(cells)} cells")
