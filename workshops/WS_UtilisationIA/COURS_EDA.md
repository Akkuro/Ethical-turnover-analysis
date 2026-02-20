# Workshop 1 — Exploratory Data Analysis (EDA) : Notions de Cours

## 1. Qu'est-ce que l'EDA ?

L'**Exploratory Data Analysis** (Analyse Exploratoire des Données) est la première étape de tout projet de Data Science / Machine Learning. Elle consiste à :

- **Comprendre la structure** du jeu de données (dimensions, types de variables, valeurs manquantes)
- **Détecter les anomalies** (outliers, incohérences)
- **Visualiser les distributions** et les relations entre variables
- **Préparer les données** pour l'alimentation d'un algorithme de ML

> L'EDA a été popularisée par le statisticien John Tukey dans les années 1970.

---

## 2. Bibliothèques Python essentielles

| Bibliothèque | Rôle |
|---|---|
| **pandas** | Manipulation et analyse de données tabulaires (DataFrame, Series) |
| **numpy** | Calcul numérique, tableaux multidimensionnels |
| **matplotlib** | Visualisation statique (graphiques de base) |
| **seaborn** | Visualisation statistique haut niveau (basée sur matplotlib) |
| **missingno** | Visualisation des données manquantes |
| **plotly** | Graphiques interactifs (3D, cartes, etc.) |
| **folium** | Cartes interactives (Leaflet.js) |

---

## 3. Chargement et exploration initiale

### Fonctions clés pandas

- `pd.read_csv(path)` : charger un fichier CSV dans un DataFrame
- `df.head(n)` : afficher les *n* premières lignes
- `df.info()` : résumé concis (types, non-null counts)
- `df.describe()` : statistiques descriptives (mean, std, min, max, quartiles)
- `df.shape` : dimensions (lignes, colonnes)
- `df.dtypes` : types de chaque colonne

### Types de variables

- **Numériques continues** : `float64` — *ex : median_income, latitude*
- **Numériques discrètes** : `int64` — *ex : housing_median_age*
- **Catégorielles** : `object` — *ex : ocean_proximity*

---

## 4. Valeurs manquantes

### Détection

```python
df.isnull().sum()           # Nombre de NaN par colonne
df.info()                   # Non-null count par colonne
msno.matrix(df)             # Visualisation matricielle
```

### Stratégies d'imputation

| Stratégie | Quand l'utiliser |
|---|---|
| **Médiane** | Variables numériques avec outliers (robuste) |
| **Moyenne** | Variables numériques à distribution symétrique |
| **Mode** | Variables catégorielles |
| **Suppression** | Si très peu de lignes concernées |
| **KNN Imputer** | Pour des imputations plus sophistiquées |

```python
# Imputation médiane avec pandas
df[col].fillna(df[col].median(), inplace=True)

# Imputation mode
df[col].fillna(df[col].mode().iloc[0], inplace=True)
```

---

## 5. Analyse univariée

L'analyse univariée étudie **une seule variable à la fois** pour comprendre sa distribution.

### Variables numériques

- **Histogramme** (`sns.histplot`) : distribution de fréquence
- **KDE** (Kernel Density Estimate) : estimation de la densité de probabilité
- **Boxplot** : médiane, quartiles, outliers

### Variables catégorielles

- **Countplot** (`sns.countplot`) : fréquence de chaque catégorie
- **value_counts()** : comptage des occurrences

### Concepts statistiques clés

- **Moyenne** ($\bar{x} = \frac{1}{n}\sum x_i$) : centre de gravité des données
- **Médiane** : valeur centrale (robuste aux outliers)
- **Écart-type** ($\sigma$) : dispersion autour de la moyenne
- **Quartiles** (Q1, Q2, Q3) : division en 4 parties égales
- **IQR** = Q3 - Q1 : intervalle interquartile

---

## 6. Analyse bivariée

L'analyse bivariée étudie **la relation entre deux variables**.

### Numérique vs Numérique

- **Scatter plot** (`plt.scatter`, `sns.scatterplot`) : points dans un plan 2D
- **Pairplot** (`sns.pairplot`) : matrice de scatter plots pour toutes les paires
- **Corrélation de Pearson** ($r$) :
  $$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$
  - $r = 1$ : corrélation positive parfaite
  - $r = -1$ : corrélation négative parfaite
  - $r = 0$ : pas de corrélation linéaire

### Catégorielle vs Numérique

- **Boxplot** (`sns.boxplot`) : compare les distributions par catégorie

### Heatmap de corrélation

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

Permet de visualiser rapidement toutes les corrélations linéaires dans un seul graphique.

---

## 7. Visualisation géographique

Quand les données contiennent latitude/longitude :

### Matplotlib — Hexbin plot

```python
plt.hexbin(df['longitude'], df['latitude'], C=df['target'], gridsize=50)
```

Agrège les données en hexagones et affiche la moyenne de la variable cible.

### Plotly — Density Mapbox

Carte interactive avec densité colorée sur fond de carte.

### Folium — Heatmap & MarkerCluster

- **HeatMap** : carte de chaleur basée sur la densité des points
- **MarkerCluster** : clusters de marqueurs qui se regroupent selon le zoom

---

## 8. Préparation des données pour le ML

L'EDA prépare le terrain pour les étapes suivantes :

1. **Nettoyage** : gestion des valeurs manquantes et des outliers
2. **Feature engineering** : création de nouvelles variables (ratios, agrégations)
3. **Encodage** : transformation des variables catégorielles (One-Hot Encoding)
4. **Normalisation / Standardisation** : mise à l'échelle des variables numériques
5. **Train/Test split** : séparation des données pour l'entraînement et l'évaluation

---

## 9. Points clés à retenir

- Toujours commencer par **explorer visuellement** les données avant de modéliser
- Les **valeurs manquantes** doivent être traitées (imputation ou suppression)
- Les **outliers** peuvent fortement biaiser les modèles → les détecter via boxplots/IQR
- La **corrélation** ne signifie pas **causalité**
- Les variables avec des **échelles très différentes** doivent être normalisées
- Les variables **catégorielles** doivent être encodées numériquement pour le ML
