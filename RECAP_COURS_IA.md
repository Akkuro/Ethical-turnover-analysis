# Bloc IA — Récapitulatif Complet des Cours

> **Acquis d'apprentissage visés** — Ce document synthétise l'ensemble des notions abordées dans les 3 boucles PBL et le projet, organisé par thématique et niveau taxonomique (Bloom). Une section finale est dédiée à la préparation de l'étude de cas.

---

## Table des matières

1. [Environnement & Outils Python](#1-environnement--outils-python)
2. [Analyse Exploratoire des Données (EDA)](#2-analyse-exploratoire-des-données-eda)
3. [Préparation des Données & Feature Engineering](#3-préparation-des-données--feature-engineering)
4. [Régression (Apprentissage Supervisé — Variable Continue)](#4-régression-apprentissage-supervisé--variable-continue)
5. [Classification (Apprentissage Supervisé — Variable Discrète)](#5-classification-apprentissage-supervisé--variable-discrète)
6. [Validation & Évaluation des Modèles](#6-validation--évaluation-des-modèles)
7. [Éthique de l'IA](#7-éthique-de-lia)
8. [Préparation à l'Étude de Cas (Examen)](#8-préparation-à-létude-de-cas-examen)

---

## 1. Environnement & Outils Python

### 1.1 Configuration de l'environnement

| Étape | Description |
|---|---|
| Installation de Python | Via Anaconda, miniconda ou `pyenv` |
| Gestionnaire de paquets | `pip`, `conda` ou `uv` |
| Environnement virtuel | `venv`, `conda env` — isole les dépendances |
| Jupyter Notebook | Interface interactive pour le prototypage et la visualisation |

**Commandes essentielles** :
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
jupyter notebook   # Lancer le serveur Jupyter
```

### 1.2 Bibliothèques fondamentales

| Bibliothèque | Rôle |
|---|---|
| **pandas** | Manipulation de données tabulaires (DataFrame, Series) |
| **numpy** | Calcul numérique, tableaux multidimensionnels |
| **matplotlib** | Visualisation statique (graphiques de base) |
| **seaborn** | Visualisation statistique haut niveau |
| **scikit-learn** | Algorithmes de ML, preprocessing, métriques, pipelines |
| **plotly / folium** | Visualisation interactive, cartes géographiques |
| **missingno** | Visualisation des données manquantes |
| **statsmodels** | Inférence statistique, p-values, résumés détaillés |

---

## 2. Analyse Exploratoire des Données (EDA)

### Acquis d'apprentissage visés
- [3] Appliquer les techniques de préparation des données pour une tâche de ML
- [3] Réaliser une EDA avec Python pour mieux comprendre un jeu de données
- [4] Utiliser les techniques d'EDA pour identifier des tendances et des corrélations
- [6] Concevoir un pipeline d'analyse exploratoire intégrant statistiques descriptives et prétraitements

### 2.1 Définition

L'**EDA** (Exploratory Data Analysis) est la première étape de tout projet de Data Science. Elle consiste à :
- Comprendre la **structure** du jeu de données (dimensions, types, valeurs manquantes)
- **Détecter les anomalies** (outliers, incohérences, doublons)
- **Visualiser** les distributions et les relations entre variables
- **Préparer** les données pour l'alimentation d'un algorithme de ML

> L'EDA a été popularisée par le statisticien **John Tukey** dans les années 1970.

### 2.2 Exploration initiale

| Fonction pandas | Description |
|---|---|
| `pd.read_csv(path)` | Charger un fichier CSV |
| `df.head(n)` | Afficher les n premières lignes |
| `df.info()` | Types, non-null counts, mémoire |
| `df.describe()` | Statistiques descriptives (mean, std, min, Q1, Q2, Q3, max) |
| `df.shape` | Dimensions (lignes, colonnes) |
| `df.dtypes` | Types de chaque colonne |
| `df.isnull().sum()` | Nombre de NaN par colonne |
| `df.duplicated().sum()` | Nombre de lignes dupliquées |
| `df.value_counts()` | Comptage par valeur (pour les catégorielles) |

### 2.3 Types de variables

| Type | Python dtype | Exemples |
|---|---|---|
| Numérique continue | `float64` | Revenu médian, latitude, prix |
| Numérique discrète | `int64` | Âge médian, nombre de pièces |
| Catégorielle | `object` | Proximité océan, genre, statut marital |
| Booléenne | `bool` / `int` | Vendu oui/non (Sold6M) |

### 2.4 Analyse univariée

Étude d'**une seule variable** à la fois :

| Variable | Visualisation | Outil |
|---|---|---|
| Numérique | Histogramme, KDE, Boxplot | `sns.histplot()`, `sns.boxplot()` |
| Catégorielle | Countplot, barplot | `sns.countplot()`, `df[col].value_counts().plot.bar()` |

**Concepts statistiques clés** :
- **Moyenne** : $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ — centre de gravité
- **Médiane** : valeur centrale, robuste aux outliers
- **Écart-type** : $\sigma = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$ — dispersion
- **Quartiles** : Q1 (25%), Q2 (50% = médiane), Q3 (75%)
- **IQR** = Q3 − Q1 — intervalle interquartile
- **Outliers** : valeurs en dehors de $[Q1 - 1.5 \times IQR, \ Q3 + 1.5 \times IQR]$

### 2.5 Analyse bivariée

Étude de la **relation entre deux variables** :

| Combinaison | Visualisation | Mesure |
|---|---|---|
| Num vs Num | Scatter plot, pairplot | Corrélation de Pearson |
| Cat vs Num | Boxplot, violinplot | Comparaison des distributions |
| Cat vs Cat | Heatmap croisé | Tableau de contingence |

**Corrélation de Pearson** :
$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \cdot \sum(y_i - \bar{y})^2}}$$

| Valeur | Interprétation |
|---|---|
| $r = 1$ | Corrélation positive parfaite |
| $r = -1$ | Corrélation négative parfaite |
| $r = 0$ | Pas de corrélation linéaire |

> ⚠️ **Corrélation ≠ Causalité** : deux variables corrélées ne signifie pas que l'une cause l'autre.

**Heatmap de corrélation** :
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### 2.6 Visualisation géographique

Quand les données contiennent latitude/longitude :
- **Hexbin plot** (matplotlib) : agrège les données en hexagones
- **Density Mapbox** (plotly) : carte interactive avec densité colorée
- **HeatMap / MarkerCluster** (folium) : carte de chaleur sur fond Leaflet

---

## 3. Préparation des Données & Feature Engineering

### Acquis d'apprentissage visés
- [4] Détecter les valeurs manquantes et appliquer les stratégies d'imputation
- [3] Mettre en œuvre la normalisation et la standardisation des données
- [6] Concevoir un pipeline de préparation des données

### 3.1 Gestion des valeurs manquantes

| Stratégie | Quand l'utiliser | Code |
|---|---|---|
| **Médiane** | Variables numériques avec outliers | `SimpleImputer(strategy="median")` |
| **Moyenne** | Variables numériques symétriques | `SimpleImputer(strategy="mean")` |
| **Mode** | Variables catégorielles | `SimpleImputer(strategy="most_frequent")` |
| **Suppression** | Très peu de lignes concernées | `df.dropna()` |
| **KNN Imputer** | Imputation sophistiquée basée sur les voisins | `KNNImputer(n_neighbors=5)` |

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(df_num)
df_num_imputed = pd.DataFrame(imputer.transform(df_num), columns=df_num.columns)
```

### 3.2 Encodage des variables catégorielles

#### One-Hot Encoding
Transforme chaque catégorie en colonne binaire :

| ocean_proximity | → INLAND | → NEAR BAY | → NEAR OCEAN |
|---|---|---|---|
| INLAND | 1 | 0 | 0 |
| NEAR BAY | 0 | 1 | 0 |

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
df_cat_encoded = pd.DataFrame(
    encoder.fit_transform(df_cat),
    columns=encoder.get_feature_names_out(df_cat.columns)
)
```

#### Label Encoding
Attribue un entier à chaque catégorie (à utiliser pour les variables **ordinales**) :
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col'] = le.fit_transform(df['col'])
```

### 3.3 Normalisation et Standardisation

#### Standardisation (StandardScaler) — Centrer-réduire
$$x_{std} = \frac{x - \mu}{\sigma}$$

Résultat : $\mu = 0$, $\sigma = 1$. **Indispensable** pour SVM, KNN, Perceptron, Régression Logistique.

#### Min-Max Scaling (Normalisation)
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Résultat : valeurs dans $[0, 1]$. Utile quand on veut borner les valeurs.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
```

### 3.4 Pipeline scikit-learn

Un **pipeline** chaîne des transformations avec un estimateur final :

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Pipeline numérique
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
])

# Pipeline complet
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

X_prepared = full_pipeline.fit_transform(X)
```

**Avantages** :
- Reproductibilité garantie
- Pas de **data leakage** (fit uniquement sur le train set)
- Code propre et maintenable

### 3.5 Séparation Train / Test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Split stratifié
Garantit les mêmes proportions de classes dans chaque sous-ensemble :
```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

---

## 4. Régression (Apprentissage Supervisé — Variable Continue)

### Acquis d'apprentissage visés
- [1] Identifier une tâche de régression à partir d'un jeu de données
- [1] Lister les différents types de régression (simple, multiple, polynomiale, par noyau…)
- [4] Analyser les performances avec R² et MSE
- [5] Estimer la pertinence et l'efficacité des modèles développés
- [6] Proposer des améliorations en fonction des résultats et des analyses de résidus

### 4.1 Définition

La **régression** prédit une **variable continue** (quantitative) à partir de variables explicatives.

**Exemples** : prix d'un logement, salaire, chiffre d'affaires.

### 4.2 Régression Linéaire

#### Modèle
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$

En forme matricielle : $\hat{Y} = X\theta$

#### Méthode des Moindres Carrés Ordinaires (OLS)

**Objectif** : minimiser la somme des résidus au carré :
$$\min_{\theta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**Solution analytique — Équation Normale** :
$$\hat{\theta} = (X^T X)^{-1} X^T Y$$

- **Avantages** : solution exacte, pas d'itérations
- **Inconvénients** : coûteux pour de très grands datasets ($O(n^3)$)

#### Interprétation des coefficients
- $\theta_i > 0$ : quand $x_i$ augmente, $\hat{y}$ augmente
- $\theta_i < 0$ : quand $x_i$ augmente, $\hat{y}$ diminue
- **p-value < 0.05** : le coefficient est statistiquement significatif

### 4.3 Arbre de Décision pour la Régression

Le `DecisionTreeRegressor` partitionne l'espace des features en rectangles et prédit la **valeur moyenne** de chaque partition.

⚠️ **Risque de sur-apprentissage** : un arbre profond peut mémoriser les données d'entraînement (RMSE = 0 sur le train set) mais généraliser mal. Solution : limiter `max_depth`, `min_samples_leaf`.

### 4.4 Métriques de performance en régression

| Métrique | Formule | Interprétation |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Erreur moyenne quadratique. Sensible aux outliers. |
| **RMSE** | $\sqrt{MSE}$ | Même unité que la cible → plus interprétable |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Plus robuste aux outliers |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 1 = parfait, 0 = aussi bon que la moyenne, <0 = pire |

> **Note** : on ne peut **PAS** utiliser la MSE pour évaluer un modèle de classification. La MSE est réservée à la régression (variable continue). En classification, on utilise Accuracy, Precision, Recall, F1, AUC.

### 4.5 Comparaison des modèles de régression

| Modèle | Forces | Faiblesses |
|---|---|---|
| **Régression Linéaire** | Simple, interprétable, rapide | Suppose une relation linéaire |
| **Arbre de Décision** | Capture les non-linéarités | Overfitting facile |
| **SVR** | Bon en haute dimension | Lent pour grands datasets |
| **KNN Regressor** | Simple, pas d'entraînement | Lent en prédiction, sensible à la dimension |

---

## 5. Classification (Apprentissage Supervisé — Variable Discrète)

### Acquis d'apprentissage visés
- [1] Lister les différents algorithmes de classification en ML
- [2] Définir les concepts de classification et en comprendre les principes
- [3] Appliquer des approches supervisées de classification pour prédire des catégories
- [4] Expliquer les principes de base des algorithmes (k-NN, arbres de décision, SVM)
- [4] Analyser les performances (précision, rappel, F1-score, AUC)
- [5] Utiliser la régression logistique pour prédire la probabilité d'un événement binaire
- [6] Combiner plusieurs techniques de prétraitement et d'algorithmes pour optimiser les résultats

### 5.1 Définition

La **classification** prédit une **catégorie discrète** (classe) à partir de variables explicatives.

- **Classification binaire** : 2 classes (ex : vendu dans 6 mois — oui/non, attrition — oui/non)
- **Classification multi-classes** : 3+ classes (ex : type de fleur, niveau de satisfaction)

### 5.2 Algorithmes de Classification — Fiches Détaillées

---

#### 5.2.1 Régression Logistique

**Principe** : malgré son nom, c'est un **classificateur** binaire. Utilise la **fonction sigmoïde** pour transformer une combinaison linéaire en probabilité.

**Fonction sigmoïde** :
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Modèle** :
$$P(y=1|x) = \sigma(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n)$$

**Fonction de coût** (log-vraisemblance / binary cross-entropy) :
$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\right]$$

**Entraînement** : descente de gradient pour minimiser $J(\beta)$.

| Forces | Faiblesses |
|---|---|
| Interprétable (coefficients = importance) | Frontières de décision linéaires |
| Probabilités calibrées | Mal adaptée aux relations non-linéaires |
| Rapide à entraîner | |

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

#### 5.2.2 Perceptron

**Principe** : le neurone artificiel le plus simple (Rosenblatt, 1957). Calcule une somme pondérée des entrées et applique une **fonction de seuil**.

**Sortie** :
$$\hat{y} = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{si } z < 0 \end{cases}$$

où $z = \beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n$

**Règle de mise à jour** :
$$\beta_j \leftarrow \beta_j + \eta(y^{(i)} - \hat{y}^{(i)}) x_j^{(i)}$$

$\eta$ = taux d'apprentissage (learning rate).

| Forces | Faiblesses |
|---|---|
| Extrêmement simple et rapide | Ne converge que si les données sont linéairement séparables |
| Base des réseaux de neurones | Pas de sortie probabiliste |

```python
from sklearn.linear_model import Perceptron
model = Perceptron(random_state=42)
```

---

#### 5.2.3 Support Vector Machine (SVM)

**Principe** : cherche l'**hyperplan optimal** qui maximise la **marge** entre les classes. Les points les plus proches de l'hyperplan sont appelés **vecteurs de support**.

**Optimisation** :
$$\min \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^{m}\xi_i$$

Sous les contraintes : $y^{(i)}(\beta \cdot x^{(i)} + \beta_0) \geq 1 - \xi_i$

- $C$ : paramètre de **régularisation** (haut → marge étroite, peu d'erreurs ; bas → marge large, tolérance aux erreurs)
- $\xi_i$ : variables de relâchement (soft margin)

**Noyaux (Kernels)** — pour les données non linéairement séparables :

| Noyau | Formule | Usage typique |
|---|---|---|
| Linéaire | $K(x,x') = x \cdot x'$ | Données linéairement séparables |
| Polynomial | $K(x,x') = (\gamma x \cdot x' + r)^d$ | Relations polynomiales |
| **RBF (Gaussien)** | $K(x,x') = e^{-\gamma\|x-x'\|^2}$ | Le plus courant, flexible |
| Sigmoïde | $K(x,x') = \tanh(\gamma x \cdot x' + r)$ | Similaire aux réseaux de neurones |

| Forces | Faiblesses |
|---|---|
| Excellent en haute dimension | Lent sur de grands datasets |
| Flexible via les noyaux | Sensible aux hyperparamètres ($C$, $\gamma$) |
| Efficace en mémoire (vecteurs de support) | Résultats peu interprétables |

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', probability=True)
```

---

#### 5.2.4 Naive Bayes (Gaussien)

**Principe** : basé sur le **théorème de Bayes** avec hypothèse d'**indépendance conditionnelle** entre les features.

**Théorème de Bayes** :
$$P(C_k | \mathbf{x}) = \frac{P(C_k) \cdot P(\mathbf{x} | C_k)}{P(\mathbf{x})}$$

Avec hypothèse naïve : $P(\mathbf{x} | C_k) = \prod_{i=1}^{n} P(x_i | C_k)$

**Décision** : choisir la classe $C_k$ qui maximise $P(C_k) \cdot \prod P(x_i | C_k)$.

**Variantes** :
| Variante | Hypothèse sur les features | Cas d'usage |
|---|---|---|
| **GaussianNB** | Distribution normale | Features continues |
| **MultinomialNB** | Comptage d'occurrences | NLP, texte |
| **BernoulliNB** | Binaires (0/1) | Features booléennes |

| Forces | Faiblesses |
|---|---|
| Très rapide | Hypothèse d'indépendance rarement vraie |
| Bon en baseline | Probabilités peu calibrées |
| Fonctionne bien avec peu de données | Ignore les interactions entre features |

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```

---

#### 5.2.5 K-Nearest Neighbors (KNN)

**Principe** : classe un point selon la **majorité des classes de ses $k$ voisins** les plus proches.

**Distance euclidienne** :
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Hyperparamètre principal** : $k$ (nombre de voisins).
- $k$ petit → modèle complexe, risque d'overfitting
- $k$ grand → modèle lisse, risque d'underfitting

| Forces | Faiblesses |
|---|---|
| Simple, intuitif | Lent en prédiction ($O(n)$ par prédiction) |
| Pas de phase d'entraînement | Sensible à l'échelle → normalisation obligatoire |
| Non-paramétrique | Mauvais en haute dimension (curse of dimensionality) |

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
```

---

#### 5.2.6 Arbre de Décision

**Principe** : partitionne l'espace des features par des **tests successifs** formant une structure d'arbre (nœuds de décision → feuilles = classes).

**Critères de partitionnement** :

| Critère | Formule | Description |
|---|---|---|
| **Entropie** | $H = -\sum p_i \log_2(p_i)$ | Mesure du désordre. Gain d'information = réduction d'entropie |
| **Indice de Gini** | $G = 1 - \sum p_i^2$ | Mesure d'impureté. 0 = pur, 0.5 = maximum (binaire) |

**Construction** :
1. Sélectionner la feature qui maximise le gain d'information (ou minimise Gini)
2. Diviser les données selon cette feature
3. Répéter récursivement sur chaque sous-ensemble
4. S'arrêter quand un critère est atteint (profondeur max, nœud pur, etc.)

**Hyperparamètres clés** :
- `max_depth` : profondeur maximale de l'arbre
- `min_samples_split` : nombre min d'échantillons pour diviser un nœud
- `min_samples_leaf` : nombre min d'échantillons dans une feuille

| Forces | Faiblesses |
|---|---|
| Très interprétable (visualisable) | Overfitting facile |
| Gère numériques et catégorielles | Variance élevée (sensible aux données) |
| Pas de normalisation nécessaire | Biais vers les features à nombreuses valeurs |

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
```

---

#### 5.2.7 Random Forest (Forêt Aléatoire)

**Principe** : **méthode d'ensemble** combinant plusieurs arbres de décision. Chaque arbre vote et la classe finale est déterminée par **vote majoritaire**.

**Construction** :
1. **Échantillonnage Bootstrap** : chaque arbre est entraîné sur un échantillon aléatoire avec remplacement
2. **Sélection aléatoire de features** : à chaque nœud, seul un sous-ensemble de features est considéré
3. **Vote majoritaire** : la classe finale = la classe la plus votée par tous les arbres

**Hyperparamètres** :
- `n_estimators` : nombre d'arbres (100 par défaut)
- `max_depth` : profondeur maximale de chaque arbre
- `max_features` : nombre de features par nœud (`'sqrt'` par défaut)

| Forces | Faiblesses |
|---|---|
| Réduit l'overfitting vs arbre unique | Moins interprétable |
| Robuste, peu de tuning nécessaire | Plus lent (n arbres) |
| Feature importance intégrée | Consomme plus de mémoire |

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### 5.3 Tableau comparatif des modèles

| Modèle | Complexité | Interprétabilité | Vitesse Train | Vitesse Predict | Normalisation requise |
|---|---|---|---|---|---|
| Régression Logistique | Faible | Haute | Rapide | Rapide | Oui |
| Perceptron | Faible | Haute | Très rapide | Très rapide | Oui |
| SVM | Moyenne-Haute | Faible | Lent | Moyen | Oui |
| Naive Bayes | Faible | Moyenne | Très rapide | Très rapide | Non |
| KNN | Faible | Haute | Aucun | Lent | Oui |
| Arbre de Décision | Moyenne | Haute | Rapide | Rapide | Non |
| Random Forest | Haute | Faible | Moyen | Moyen | Non |

---

## 6. Validation & Évaluation des Modèles

### 6.1 Métriques de classification

#### Matrice de confusion

$$\begin{array}{|c|c|c|}
\hline
& \text{Prédit Positif} & \text{Prédit Négatif} \\
\hline
\text{Réel Positif} & \text{VP (True Positive)} & \text{FN (False Negative)} \\
\hline
\text{Réel Négatif} & \text{FP (False Positive)} & \text{VN (True Negative)} \\
\hline
\end{array}$$

#### Métriques dérivées

| Métrique | Formule | Quand l'utiliser |
|---|---|---|
| **Accuracy** | $\frac{VP+VN}{VP+VN+FP+FN}$ | Classes équilibrées uniquement |
| **Precision** | $\frac{VP}{VP+FP}$ | Coût des FP élevé (ex : spam) |
| **Recall** | $\frac{VP}{VP+FN}$ | Coût des FN élevé (ex : maladie) |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Compromis Precision/Recall |

#### Courbe ROC et AUC

- **Courbe ROC** : trace le TPR (Recall) en fonction du FPR ($\frac{FP}{FP+VN}$) pour différents seuils
- **AUC** : aire sous la courbe ROC

| AUC | Interprétation |
|---|---|
| 0.9 – 1.0 | Excellente |
| 0.8 – 0.9 | Bonne |
| 0.7 – 0.8 | Acceptable |
| 0.6 – 0.7 | Faible |
| 0.5 – 0.6 | Très faible (≈ hasard) |

#### Courbe Precision-Recall
Utile quand les classes sont **déséquilibrées** : trace la Precision en fonction du Recall pour différents seuils.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
print(classification_report(y_test, y_pred))
```

### 6.2 Validation croisée (Cross-Validation)

**Pourquoi ?** Ne pas utiliser le jeu de test pour ajuster le modèle → estimation fiable de la généralisation.

**K-Fold Cross-Validation** :
1. Diviser les données en $k$ folds (typiquement $k = 5$ ou $10$)
2. Pour chaque fold $i$ : entraîner sur les $k-1$ autres, évaluer sur le fold $i$
3. Obtenir $k$ scores → calculer **moyenne** et **écart-type**

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(f"Moyenne: {rmse_scores.mean():.2f}, Écart-type: {rmse_scores.std():.2f}")
```

L'écart-type donne une mesure de la **stabilité** du modèle.

### 6.3 Déséquilibre de classes

Quand une classe est beaucoup plus fréquente : l'**accuracy est trompeuse**.

**Solutions** :
| Technique | Description |
|---|---|
| **Sous-échantillonnage** | Réduire la classe majoritaire |
| **Sur-échantillonnage (SMOTE)** | Générer des exemples synthétiques de la classe minoritaire |
| **Pondération** | `class_weight='balanced'` dans le modèle |
| **Métriques adaptées** | Utiliser F1-Score ou AUC au lieu de l'Accuracy |

### 6.4 Sur-apprentissage (Overfitting) vs Sous-apprentissage (Underfitting)

| Situation | Symptôme | Solutions |
|---|---|---|
| **Overfitting** | Train score excellent, Test score médiocre | Régularisation, réduire la complexité, plus de données, cross-validation |
| **Underfitting** | Train et Test scores médiocres | Modèle plus complexe, plus de features, moins de régularisation |

---

## 7. Éthique de l'IA

### Acquis d'apprentissage visés
- [4] Détecter les problématiques éthiques liées à l'IA dans un contexte donné
- [4] Critiquer les politiques et pratiques actuelles en matière d'éthique et de durabilité dans l'IA
- [5] Défendre l'importance d'une approche éthique et durable

### 7.1 Les 7 exigences de la Commission Européenne

| # | Exigence | Questions clés |
|---|---|---|
| 1 | **Respect de l'autonomie humaine** | L'humain garde-t-il le contrôle ? Le système informe-t-il l'utilisateur qu'il interagit avec une IA ? |
| 2 | **Robustesse technique et sécurité** | Le modèle est-il fiable ? Résiste-t-il aux attaques adverses ? Plan de repli si ça échoue ? |
| 3 | **Confidentialité et gouvernance des données** | Les données sont-elles anonymisées ? Consentement obtenu ? RGPD respecté ? |
| 4 | **Transparence** | Le modèle est-il explicable ? Les décisions sont-elles documentées ? |
| 5 | **Diversité, non-discrimination et équité** | Le dataset contient-il des biais (genre, ethnie, âge) ? Le modèle est-il équitable pour tous les groupes ? |
| 6 | **Bien-être environnemental et sociétal** | Impact environnemental du calcul ? Impact social (emploi, société) ? |
| 7 | **Responsabilité** | Qui est responsable des décisions du modèle ? Mécanismes d'audit ? |

### 7.2 Application au projet (HumanForYou)

- **Données sensibles** : âge, genre, statut marital, salaire → risque de discrimination
- **Anonymisation** : EmployeeID remplace les noms → respecte la confidentialité
- **Biais potentiels** : le modèle pourrait apprendre des corrélations discriminatoires (ex : genre → attrition)
- **Transparence** : privilégier des modèles interprétables (Logistic Regression, Decision Tree) ou utiliser SHAP/LIME pour expliquer les modèles complexes
- **Impact RH** : les recommandations doivent être éthiques (ne pas stigmatiser des groupes d'employés)

---

## 8. Préparation à l'Étude de Cas (Examen)

> **Format attendu** (basé sur les années précédentes) : on vous donne un **dataset** et une **problématique d'entreprise**. Vous devez :
> 1. Proposer des améliorations pour nettoyer les données
> 2. Décrire un algorithme (définition, étapes, cas concret)
> 3. Comprendre le problème d'une entreprise, proposer un algorithme adapté et le justifier

### 8.1 Partie 1 — Nettoyage et préparation d'un dataset

**Méthodologie à suivre systématiquement** :

#### Étape 1 : Exploration initiale
```python
df.info()         # Types, valeurs manquantes
df.describe()     # Statistiques descriptives
df.head()         # Aperçu des données
df.shape          # Dimensions
df.duplicated().sum()  # Doublons
```

#### Étape 2 : Identifier les problèmes
| Problème | Comment le détecter | Solution |
|---|---|---|
| **Valeurs manquantes** | `df.isnull().sum()` | Imputation (médiane, mode) ou suppression |
| **Doublons** | `df.duplicated().sum()` | `df.drop_duplicates()` |
| **Outliers** | Boxplot, IQR | Suppression, clipping, ou transformation log |
| **Types incorrects** | `df.dtypes` | `df.astype()` ou `pd.to_numeric()` |
| **Variables catégorielles** | `df.select_dtypes(include='object')` | One-Hot Encoding ou Label Encoding |
| **Échelles différentes** | `df.describe()` | StandardScaler ou MinMaxScaler |
| **Déséquilibre de classes** | `df[target].value_counts()` | SMOTE, sous-échantillonnage, pondération |
| **Multicolinéarité** | Heatmap de corrélation | Suppression d'une des variables corrélées |
| **Variables inutiles** | Analyse du domaine | Suppression (ex : ID, constantes) |

#### Étape 3 : Appliquer les corrections
- Imputer les valeurs manquantes
- Encoder les catégorielles
- Normaliser/standardiser
- Créer de nouvelles features (feature engineering)
- Séparer train/test

#### Étape 4 : Vérification
- Vérifier qu'il n'y a plus de NaN
- Vérifier les dimensions
- Visualiser les distributions après transformation

### 8.2 Partie 2 — Décrire un algorithme

**Structure attendue pour décrire un algorithme** :

#### 1. Définition
- Nom complet et type (supervisé/non-supervisé, classification/régression)
- En une phrase : que fait cet algorithme ?

#### 2. Principe de fonctionnement
- Explication intuitive (sans formules complexes)
- Explication mathématique (formules clés)

#### 3. Étapes de l'algorithme
Numéroter chaque étape clairement :
1. Étape 1 : …
2. Étape 2 : …
3. Etc.

#### 4. Hyperparamètres importants
- Lister et expliquer chaque hyperparamètre clé

#### 5. Forces et faiblesses
- Tableau forces/faiblesses

#### 6. Cas concret
- Appliquer l'algorithme à un exemple du domaine étudié

---

**Exemple — Régression Logistique** :

> **Définition** : Algorithme de classification supervisé binaire qui modélise la probabilité d'appartenance à une classe via la fonction sigmoïde.
>
> **Principe** : Combine linéairement les features ($z = \beta_0 + \beta_1 x_1 + \ldots$), puis transforme via $\sigma(z) = \frac{1}{1+e^{-z}}$ pour obtenir une probabilité entre 0 et 1. Seuil par défaut : 0.5.
>
> **Étapes** :
> 1. Initialiser les poids $\beta$ aléatoirement
> 2. Calculer $z = X\beta$ pour chaque observation
> 3. Appliquer la sigmoïde : $\hat{y} = \sigma(z)$
> 4. Calculer l'erreur via la fonction de coût (cross-entropy)
> 5. Mettre à jour les poids par descente de gradient
> 6. Répéter jusqu'à convergence
>
> **Cas concret** : Prédire si un bien immobilier sera vendu dans les 6 mois (Sold6M). Features : revenu médian, âge logement, proximité océan. Sortie : probabilité de vente. Si P > 0.5 → vendu.

---

**Exemple — Random Forest** :

> **Définition** : Méthode d'ensemble de classification/régression supervisée combinant N arbres de décision indépendants.
>
> **Étapes** :
> 1. Créer N échantillons bootstrap (tirage avec remise) du dataset
> 2. Pour chaque échantillon, construire un arbre de décision en ne considérant qu'un sous-ensemble aléatoire de features à chaque nœud
> 3. Laisser chaque arbre grandir sans élagage
> 4. Pour une prédiction : chaque arbre vote → la classe majoritaire est retenue
>
> **Cas concret** : Prédire l'attrition des employés (HumanForYou). 100 arbres entraînés chacun sur un échantillon différent. Si 73 arbres prédisent "quitte" et 27 "reste" → prédiction = "quitte l'entreprise".

### 8.3 Partie 3 — Étude de cas : entreprise + problématique

**Méthodologie pour choisir et justifier un algorithme** :

#### Étape 1 : Comprendre le problème
- **Quel est l'objectif ?** Prédire quoi ? (variable cible)
- **Quel type de problème ?**
  - Variable cible **continue** → **Régression**
  - Variable cible **catégorielle** → **Classification**
  - Pas de variable cible → **Clustering** (non supervisé)

#### Étape 2 : Analyser les données
- Combien de données ? (taille du dataset)
- Combien de features ?
- Classes équilibrées ?
- Relations linéaires ou non ?
- Besoin d'interprétabilité ?

#### Étape 3 : Arbre de décision pour le choix de l'algorithme

```
Problème → Classification binaire ?
├── OUI
│   ├── Besoin d'interprétabilité ?
│   │   ├── OUI → Régression Logistique ou Arbre de Décision
│   │   └── NON → Random Forest ou SVM
│   ├── Petit dataset ?
│   │   ├── OUI → Naive Bayes ou KNN
│   │   └── NON → Random Forest ou SVM
│   ├── Données linéairement séparables ?
│   │   ├── OUI → Régression Logistique ou Perceptron
│   │   └── NON → SVM (RBF) ou Random Forest
│   └── Classes déséquilibrées ?
│       ├── OUI → Random Forest (class_weight) + F1/AUC
│       └── NON → N'importe lequel selon le contexte
│
Problème → Régression ?
├── OUI
│   ├── Relation linéaire ?
│   │   ├── OUI → Régression Linéaire
│   │   └── NON → Arbre de Décision ou SVR
│   └── Besoin de robustesse ?
│       └── OUI → Random Forest Regressor
│
Problème → Clustering (pas de label) ?
└── OUI → K-Means, CAH
```

#### Étape 4 : Justifier votre choix

**Modèle de justification** :

> « Pour résoudre le problème de [PROBLÈME], nous proposons d'utiliser [ALGORITHME] pour les raisons suivantes :
>
> 1. **Type de problème** : il s'agit d'une tâche de [classification/régression] car la variable cible [NOM] est [continue/catégorielle].
>
> 2. **Caractéristiques des données** : le dataset contient [N] observations et [M] features. Les classes [sont/ne sont pas] équilibrées. Les features incluent des variables [numériques/catégorielles/mixtes].
>
> 3. **Choix de l'algorithme** : [ALGORITHME] est adapté car [raisons : interprétabilité requise / relations non-linéaires / robustesse à l'overfitting / vitesse / etc.].
>
> 4. **Métriques d'évaluation** : nous utiliserons [F1-Score / AUC / RMSE / R²] car [justification : classes déséquilibrées / besoin de minimiser les faux négatifs / variable continue / etc.].
>
> 5. **Pistes d'amélioration** : [optimisation des hyperparamètres (GridSearch), feature engineering, gestion du déséquilibre (SMOTE), validation croisée, etc.]. »

#### Étape 5 : Proposer des améliorations

| Axe d'amélioration | Techniques |
|---|---|
| **Optimisation des hyperparamètres** | GridSearchCV, RandomizedSearchCV |
| **Feature engineering** | Création de ratios, agrégations, interactions |
| **Sélection de features** | Feature importance (RF), corrélation, RFE |
| **Gestion du déséquilibre** | SMOTE, class_weight, sous-échantillonnage |
| **Régularisation** | L1 (Lasso) → sélection, L2 (Ridge) → réduction |
| **Modèles d'ensemble** | Bagging, Boosting (XGBoost, Gradient Boosting) |
| **Validation** | Cross-validation k-fold, courbes d'apprentissage |
| **Explicabilité** | SHAP values, LIME, feature importance |

### 8.4 Cas pratique complet : HumanForYou (Projet)

**Problématique** : L'entreprise HumanForYou a un taux de rotation de 15%. Identifier les facteurs d'attrition et proposer des modèles prédictifs.

#### Analyse du problème
- **Variable cible** : `Attrition` (oui/non) → **Classification binaire**
- **Données** : 4 fichiers CSV (general_data, manager_survey, employee_survey, in/out_time)
- **Problème supplémentaire** : probablement des classes **déséquilibrées** (85% restent, 15% partent)

#### Préparation des données
1. **Fusionner** les 4 fichiers sur `EmployeeID`
2. **Supprimer** les colonnes inutiles (`EmployeeCount`, `Over18`, `StandardHours` → constantes)
3. **Imputer** les valeurs manquantes (enquête employés : NA → mode ou médiane)
4. **Encoder** les catégorielles (BusinessTravel, Gender, JobRole, MaritalStatus, EducationField)
5. **Calculer** des features à partir des horaires (heures travaillées moyennes, variance)
6. **Normaliser** les variables numériques

#### Choix de l'algorithme
- **Random Forest** : robuste, feature importance, gère le déséquilibre
- **Régression Logistique** : interprétable, probabilités calibrées
- **Comparaison** de plusieurs modèles avec F1-Score et AUC comme métriques principales

#### Pistes d'amélioration proposées au client
- Identifier les top features d'attrition (ex : satisfaction, salaire, distance, ancienneté)
- Recommandations RH basées sur les facteurs identifiés
- Monitoring continu du modèle avec de nouvelles données

---

## Annexe : Formules Essentielles

### Régression
| Nom | Formule |
|---|---|
| Régression linéaire | $\hat{y} = \theta_0 + \theta_1 x_1 + \ldots + \theta_n x_n$ |
| Équation normale | $\hat{\theta} = (X^T X)^{-1} X^T Y$ |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |

### Classification
| Nom | Formule |
|---|---|
| Sigmoïde | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| Cross-entropy | $-\frac{1}{m}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Precision | $\frac{VP}{VP + FP}$ |
| Recall | $\frac{VP}{VP + FN}$ |
| F1-Score | $2 \times \frac{P \times R}{P + R}$ |
| Gini | $1 - \sum p_i^2$ |
| Entropie | $-\sum p_i \log_2(p_i)$ |

### Préparation
| Nom | Formule |
|---|---|
| StandardScaler | $\frac{x - \mu}{\sigma}$ |
| MinMaxScaler | $\frac{x - x_{min}}{x_{max} - x_{min}}$ |
| Pearson | $\frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$ |
