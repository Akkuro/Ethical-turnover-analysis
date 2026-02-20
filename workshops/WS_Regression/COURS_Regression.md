# Workshop 2 — Régression : Notions de Cours

## 1. Introduction à la Régression

La **régression** est une technique de Machine Learning supervisé dont l'objectif est de **prédire une variable continue** (quantitative) à partir de variables explicatives.

**Exemples** : prédiction du prix d'un logement, estimation d'un salaire, prévision de ventes.

---

## 2. Régression Linéaire

### Principe

La régression linéaire modélise la relation entre une variable dépendante $y$ et une ou plusieurs variables indépendantes $x_1, x_2, ..., x_n$ par une fonction linéaire :

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

- $\hat{y}$ : valeur prédite
- $\theta_0$ : ordonnée à l'origine (intercept/biais)
- $\theta_1, ..., \theta_n$ : coefficients (poids) des variables

### Forme matricielle

$$\hat{Y} = X\theta$$

où $X$ est la matrice des observations (avec colonne de biais) et $\theta$ le vecteur des paramètres.

---

## 3. Méthode des Moindres Carrés Ordinaires (OLS)

### Objectif

Minimiser la somme des résidus au carré :

$$\min_{\theta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

### Solution analytique — Équation Normale

$$\hat{\theta} = (X^T X)^{-1} X^T Y$$

**Conditions** : $X$ doit être de rang plein (pas de multicolinéarité parfaite).

**Avantages** : solution exacte, pas d'itérations.  
**Inconvénients** : coûteux en calcul pour de très grands datasets (inversion de matrice en $O(n^3)$).

---

## 4. Pipeline de prétraitement (scikit-learn)

### Concept

Un **pipeline** chaîne des transformations de données avec un estimateur final, garantissant la reproductibilité :

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
])
```

### Étapes typiques

1. **Imputation** (`SimpleImputer`) : remplacer les valeurs manquantes
2. **Feature Engineering** (`CombinedAttributesAdder`) : créer de nouvelles variables
3. **Standardisation** (`StandardScaler`) : centrer-réduire ($\mu=0$, $\sigma=1$)
4. **Encodage** (`OneHotEncoder`) : transformer les catégories en indicateurs binaires

### ColumnTransformer

Permet d'appliquer des transformations différentes aux colonnes numériques et catégorielles :

```python
ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
```

---

## 5. Métriques de performance en régression

### MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Pénalise fortement les grandes erreurs (résidus² )
- Sensible aux outliers
- Unité : carré de l'unité cible

### RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Même unité que la variable cible → plus interprétable

### R² — Coefficient de détermination

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $R^2 = 1$ : modèle parfait
- $R^2 = 0$ : modèle aussi bon qu'une prédiction constante (la moyenne)
- $R^2 < 0$ : modèle pire que la moyenne

### MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Plus robuste aux outliers que la MSE

---

## 6. Bibliothèques pour la régression

### statsmodels

```python
import statsmodels.api as sm
X = sm.add_constant(X)     # Ajouter l'intercept
model = sm.OLS(y, X).fit()
print(model.summary())     # Résumé détaillé (R², p-values, IC, etc.)
```

**Avantages** : résumé statistique complet, p-values des coefficients, intervalles de confiance.

### scikit-learn

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

**Avantages** : API uniforme (fit/predict), intégration avec pipelines et cross-validation.

### Comparaison

| Aspect | statsmodels | scikit-learn |
|---|---|---|
| Focus | Inférence statistique | Prédiction / ML |
| P-values | ✅ | ❌ |
| Pipeline | ❌ | ✅ |
| Cross-validation | ❌ | ✅ |

---

## 7. Interprétation des coefficients

- **Coefficient positif** ($\theta_i > 0$) : quand $x_i$ augmente, $\hat{y}$ augmente
- **Coefficient négatif** ($\theta_i < 0$) : quand $x_i$ augmente, $\hat{y}$ diminue
- **P-value** : probabilité d'observer la valeur du coefficient si l'hypothèse nulle ($\theta_i = 0$) est vraie
  - $p < 0.05$ : le coefficient est **statistiquement significatif**
  - $p \geq 0.05$ : pas de preuve suffisante que la variable a un effet

---

## 8. Arbre de Décision pour la Régression

### Principe

Le `DecisionTreeRegressor` partitionne l'espace des features en rectangles et prédit la valeur moyenne de chaque partition.

### Problème du sur-apprentissage (Overfitting)

- Un arbre profond peut **mémoriser** les données d'entraînement → RMSE = 0 sur le training set
- Mais **généralise mal** sur des données nouvelles
- Solution : limiter la profondeur (`max_depth`), le nombre minimal de samples par feuille, etc.

---

## 9. Validation Croisée (Cross-Validation)

### Pourquoi ?

- On ne veut pas utiliser le jeu de test pour ajuster le modèle
- On divise le jeu d'entraînement en $k$ sous-ensembles (folds)

### K-Fold Cross-Validation

1. Diviser les données en $k$ folds
2. Pour chaque fold $i$ :
   - Entraîner sur les $k-1$ autres folds
   - Évaluer sur le fold $i$
3. Obtenir $k$ scores → calculer moyenne et écart-type

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
```

> scikit-learn attend une **mesure de performance** (plus = mieux), d'où le `neg_mean_squared_error`.

### Avantages

- Estimation fiable de la performance de généralisation
- L'écart-type donne une mesure de la **stabilité** du modèle

---

## 10. Comparaison de modèles de régression

| Modèle | Forces | Faiblesses |
|---|---|---|
| **Régression Linéaire** | Simple, interprétable, rapide | Suppose une relation linéaire |
| **Arbre de Décision** | Capture les non-linéarités, pas de normalisation nécessaire | Overfitting facile |
| **SVR** (Support Vector Regression) | Bon avec des features en haute dimension | Lent pour de grands datasets |
| **KNN** | Simple, pas d'entraînement | Lent en prédiction, sensible à la dimension |
| **Naive Bayes** | Très rapide | Conçu pour la classification, approximatif en régression |

---

## 11. Train-Test Split stratifié

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, X["income_cat"]):
    train_set = X.iloc[train_index]
    test_set = X.iloc[test_index]
```

Le split **stratifié** garantit que chaque sous-ensemble a la même proportion de catégories que l'ensemble original, évitant les biais d'échantillonnage.

---

## 12. Points clés à retenir

1. La régression linéaire OLS donne une **solution analytique exacte** via l'équation normale
2. Un modèle avec une erreur parfaite sur le training set est probablement en **overfitting**
3. La **validation croisée** est indispensable pour estimer la capacité de généralisation
4. Le **R²** et le **RMSE** sont les métriques principales en régression
5. Les **pipelines** garantissent la reproductibilité et évitent les fuites de données (data leakage)
6. Toujours comparer plusieurs modèles et choisir le meilleur via des métriques objectives
