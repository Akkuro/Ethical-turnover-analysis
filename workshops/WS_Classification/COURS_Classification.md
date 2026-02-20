# Workshop 3 — Classification : Notions de Cours

## 1. Introduction à la Classification

La **classification** est une technique de ML supervisé dont l'objectif est de **prédire une catégorie discrète** (classe) à partir de variables explicatives.

- **Classification binaire** : 2 classes (ex : vendu dans 6 mois oui/non)
- **Classification multi-classes** : 3+ classes (ex : type de fleur)

---

## 2. Préparation des données pour la classification

### 2.1 Gestion des valeurs manquantes

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
housing_num_imputed = pd.DataFrame(imputer.transform(housing_num), columns=housing_num.columns)
```

### 2.2 Encodage des variables catégorielles — One-Hot Encoding

Transforme chaque catégorie en colonne binaire (0 ou 1) :

| ocean_proximity | → | INLAND | NEAR BAY | NEAR OCEAN | ... |
|---|---|---|---|---|---|
| INLAND | | 1 | 0 | 0 | ... |
| NEAR BAY | | 0 | 1 | 0 | ... |

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
housing_cat_encoded = pd.DataFrame(
    encoder.fit_transform(housing_cat),
    columns=encoder.get_feature_names_out(housing_cat.columns)
)
```

### 2.3 Normalisation (StandardScaler)

$$x_{norm} = \frac{x - \mu}{\sigma}$$

Centre ($\mu = 0$) et réduit ($\sigma = 1$) les variables. Essentiel pour les algorithmes sensibles à l'échelle (SVM, KNN, Perceptron, Régression Logistique).

### 2.4 Séparation Train / Test

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 3. Métriques de Performance en Classification

### 3.1 Matrice de Confusion

$$\begin{array}{|c|c|c|}
\hline
& \text{Prédit Positif} & \text{Prédit Négatif} \\
\hline
\text{Réel Positif} & \text{VP (True Positive)} & \text{FN (False Negative)} \\
\hline
\text{Réel Négatif} & \text{FP (False Positive)} & \text{VN (True Negative)} \\
\hline
\end{array}$$

### 3.2 Accuracy (Exactitude)

$$\text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}$$

Proportion de prédictions correctes. **Trompeuse** si les classes sont déséquilibrées.

### 3.3 Precision (Précision)

$$\text{Precision} = \frac{VP}{VP + FP}$$

Parmi les prédictions positives, combien sont correctes ? Utile quand le **coût des faux positifs** est élevé.

### 3.4 Recall (Rappel / Sensibilité)

$$\text{Recall} = \frac{VP}{VP + FN}$$

Parmi les vrais positifs, combien ont été détectés ? Utile quand le **coût des faux négatifs** est élevé.

### 3.5 F1-Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Moyenne harmonique de la Precision et du Recall. Bon compromis quand les classes sont déséquilibrées.

### 3.6 Courbe ROC et AUC

- **Courbe ROC** : trace le **TPR** (Recall) en fonction du **FPR** ($\frac{FP}{FP+VN}$) pour différents seuils
- **AUC** (Area Under Curve) :
  - $AUC = 1.0$ : modèle parfait
  - $AUC = 0.5$ : modèle aléatoire
  - Interprétation : probabilité que le modèle classe un positif au-dessus d'un négatif

| AUC | Interprétation |
|---|---|
| 0.9 - 1.0 | Excellente |
| 0.8 - 0.9 | Bonne |
| 0.7 - 0.8 | Acceptable |
| 0.6 - 0.7 | Faible |
| 0.5 - 0.6 | Très faible |

---

## 4. Modèles de Classification

### 4.1 Régression Logistique

Malgré son nom, c'est un **classificateur** ! Elle utilise la **fonction sigmoïde** pour transformer une combinaison linéaire en probabilité :

$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

où $z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$

**Fonction de coût** (log-vraisemblance / cross-entropy) :

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

**Forces** : interprétable, probabilités calibrées, rapide.  
**Faiblesses** : suppose des frontières de décision linéaires.

### 4.2 Perceptron

Le modèle de neurone le plus simple (Rosenblatt, 1957) :

$$\hat{y} = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{si } z < 0 \end{cases}$$

**Mise à jour des poids** :
$$\beta_j = \beta_j + \eta(y^{(i)} - \hat{y}^{(i)})x_j^{(i)}$$

où $\eta$ est le **taux d'apprentissage** (learning rate).

**Forces** : extrêmement simple et rapide.  
**Faiblesses** : ne converge que si les données sont linéairement séparables, pas de probabilités.

### 4.3 Support Vector Machine (SVM)

Cherche l'**hyperplan optimal** qui maximise la **marge** entre les classes.

$$\text{Minimiser} \quad \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^{m}\xi_i$$

- $C$ : paramètre de **régularisation** (compromis marge/erreurs)
- $\xi_i$ : variables de relâchement (soft margin)
- **Vecteurs de support** : points les plus proches de la frontière

#### Noyaux (Kernels)

Permettent de traiter des données **non linéairement séparables** en les projetant dans un espace de dimension supérieure :

| Noyau | Formule | Usage |
|---|---|---|
| Linéaire | $K(x,x') = x \cdot x'$ | Données linéairement séparables |
| Polynomial | $K(x,x') = (\gamma x \cdot x' + r)^d$ | Relations polynomiales |
| RBF (Gaussien) | $K(x,x') = e^{-\gamma\|x-x'\|^2}$ | Le plus courant, flexibilité maximale |
| Sigmoïde | $K(x,x') = \tanh(\gamma x \cdot x' + r)$ | Similaire au réseau de neurones |

**Forces** : excellent en haute dimension, flexible via les noyaux.  
**Faiblesses** : lent sur de grands datasets, sensible aux hyperparamètres.

### 4.4 Naive Bayes (Gaussien)

Basé sur le **théorème de Bayes** avec hypothèse d'**indépendance conditionnelle** :

$$P(C_k | \mathbf{x}) = \frac{P(C_k) \cdot \prod_{i=1}^{n} P(x_i | C_k)}{P(\mathbf{x})}$$

Le classificateur choisit la classe qui maximise $P(C_k)\prod P(x_i|C_k)$.

**Variantes** :
- **GaussianNB** : features continues (distribution normale)
- **MultinomialNB** : comptage de mots (NLP)
- **BernoulliNB** : features binaires

**Forces** : très rapide, bon en baseline, fonctionne bien avec peu de données.  
**Faiblesses** : hypothèse d'indépendance rarement vraie, probabilités peu calibrées.

### 4.5 K-Nearest Neighbors (KNN)

Classe un point selon la **majorité des classes de ses k voisins les plus proches**.

- Distance euclidienne : $d(x,y) = \sqrt{\sum(x_i - y_i)^2}$
- **Hyperparamètre** : $k$ (nombre de voisins)

**Forces** : simple, pas d'entraînement.  
**Faiblesses** : lent en prédiction ($O(n)$), sensible aux dimensions et à l'échelle.

### 4.6 Arbre de Décision

Partitionne l'espace des features par des **tests successifs** formant un arbre.

**Critères de partitionnement** :
- **Entropie** : $H = -\sum p_i \log_2(p_i)$ → Gain d'information
- **Indice de Gini** : $G = 1 - \sum p_i^2$ → Mesure d'impureté

**Forces** : interprétable, gère numériques et catégorielles.  
**Faiblesses** : overfitting (contrôler `max_depth`, `min_samples_leaf`).

### 4.7 Random Forest (Forêt Aléatoire)

**Méthode d'ensemble** combinant plusieurs arbres de décision :

1. **Bootstrap** : chaque arbre est entraîné sur un échantillon aléatoire avec remplacement
2. **Random feature selection** : chaque nœud utilise un sous-ensemble aléatoire de features
3. **Vote majoritaire** : la classe finale est déterminée par le vote de tous les arbres

**Hyperparamètres** :
- `n_estimators` : nombre d'arbres
- `max_depth` : profondeur maximale de chaque arbre
- `max_features` : nombre de features considérées par nœud

**Forces** : réduit l'overfitting, robuste, feature importance.  
**Faiblesses** : moins interprétable, plus lent qu'un seul arbre.

---

## 5. Comparaison des modèles

| Modèle | Complexité | Interprétabilité | Vitesse Train | Vitesse Predict |
|---|---|---|---|---|
| Régression Logistique | Faible | Haute | Rapide | Rapide |
| Perceptron | Faible | Haute | Très rapide | Très rapide |
| SVM | Moyenne-Haute | Faible | Lent | Moyen |
| Naive Bayes | Faible | Moyenne | Très rapide | Très rapide |
| KNN | Faible | Haute | Aucun | Lent |
| Arbre de Décision | Moyenne | Haute | Rapide | Rapide |
| Random Forest | Haute | Faible | Moyen | Moyen |

---

## 6. Déséquilibre de classes

Quand une classe est beaucoup plus fréquente que l'autre, l'accuracy peut être trompeuse.

**Solutions** :
- **Sous-échantillonnage** (undersampling) de la classe majoritaire
- **Sur-échantillonnage** (oversampling) de la classe minoritaire (ex : SMOTE)
- **Pondération des classes** (`class_weight='balanced'`)
- Utiliser **F1-Score** ou **AUC** plutôt que l'accuracy

---

## 7. Workflow scikit-learn

```python
# 1. Créer le modèle
model = LogisticRegression(max_iter=1000)

# 2. Entraîner
model.fit(X_train, y_train)

# 3. Prédire
y_pred = model.predict(X_test)

# 4. Évaluer
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))
```

---

## 8. Points clés à retenir

1. Choisir les **métriques appropriées** selon le contexte (Precision vs Recall vs F1)
2. La **matrice de confusion** est l'outil de base pour comprendre les erreurs
3. La **courbe ROC / AUC** permet de comparer les modèles indépendamment du seuil
4. **Normaliser les données** est crucial pour SVM, KNN et Perceptron
5. **Random Forest** est souvent un excellent point de départ (robuste, peu de tuning)
6. Toujours **comparer plusieurs modèles** et analyser leurs temps d'entraînement/prédiction
7. Le **déséquilibre de classes** doit être traité avant la modélisation
