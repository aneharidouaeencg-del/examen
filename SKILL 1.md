

# Prompt Structuré pour Générer un Compte Rendu Académique

---

## 📋 PROMPT COMPLET À UTILISER

---

```
Tu es un assistant académique expert en data science appliquée à la supply chain.
Génère un compte rendu académique COMPLET et PROFESSIONNEL pour un projet
de fin de semestre, en respectant STRICTEMENT la structure et les consignes ci-dessous.

============================================================
PARTIE 0 : PAGE DE GARDE
============================================================

Génère une page de garde académique contenant :
- Logo placeholder : [Logo ENCG Settat] | [Logo Université Hassan 1er]
- Intitulé : "Université Hassan 1er - École Nationale de Commerce et de Gestion de Settat"
- Filière : "Purchasing and Supply Chain Management"
- Semestre : S8
- Titre du projet : "Application des techniques de Machine Learning pour la
  classification/prédiction dans le domaine de la Supply Chain"
  (ou un titre plus spécifique selon le sujet choisi ci-dessous)
- Mention : "Compte rendu de projet - Module : [Data Science / Intelligence
  Artificielle / Big Data] (choisis le plus pertinent)"
- Réalisé par : [Nom(s) de(s) l'étudiant(s)]
- Encadré par : [Nom du professeur]
- Année universitaire : 2024 - 2025
- Formater cette page de manière centrée et professionnelle.

============================================================
PARTIE 1 : SOMMAIRE
============================================================

Génère un sommaire numéroté et hiérarchisé avec numéros de pages fictifs
mais cohérents. Il doit contenir :

  Introduction générale
  Chapitre 1 : Présentation du sujet
    1.1 Contexte et problématique
    1.2 Objectifs du projet
    1.3 Revue de littérature (état de l'art)
  Chapitre 2 : Contractualisation du projet
    2.1 Cadre institutionnel (ENCG Settat, S8, filière PSCM)
    2.2 Périmètre et délimitation du sujet
    2.3 Méthodologie adoptée
    2.4 Outils et technologies utilisés
    2.5 Planning et répartition des tâches
  Chapitre 3 : Développement technique - Script Python
    3.1 Présentation du dataset utilisé (données réelles)
    3.2 Prétraitement des données (nettoyage, transformation, feature engineering)
    3.3 Analyse exploratoire des données (EDA)
    3.4 Construction du modèle de classification/prédiction
    3.5 Code Python complet et commenté
  Chapitre 4 : Analyse des résultats
    4.1 Métriques d'évaluation (accuracy, precision, recall, F1-score, AUC...)
    4.2 Interprétation des résultats
    4.3 Comparaison de modèles (si applicable)
    4.4 Discussion et limites
  Chapitre 5 : Illustrations et visualisations
    5.1 Graphiques exploratoires
    5.2 Matrice de confusion
    5.3 Courbe ROC
    5.4 Feature importance
    5.5 Autres visualisations pertinentes
  Conclusion générale et perspectives
  Bibliographie / Webographie
  Annexes

============================================================
PARTIE 2 : INTRODUCTION GÉNÉRALE (1 page)
============================================================

Rédige une introduction générale qui :
- Contextualise le Machine Learning dans la Supply Chain Management
- Présente l'importance de la classification/prédiction pour les décisions
  d'achat et de gestion logistique
- Annonce la problématique choisie
- Présente le plan du compte rendu
- Ton : académique, formel, structuré

============================================================
PARTIE 3 : CHAPITRE 1 - PRÉSENTATION DU SUJET (3-4 pages)
============================================================

CHOISIS UN sujet parmi les suivants (ou propose le plus pertinent
pour la filière Purchasing & Supply Chain Management) :

  Option A : Prédiction des retards de livraison fournisseurs
  Option B : Classification des fournisseurs (fiables vs à risque)
  Option C : Prédiction de la demande de produits
  Option D : Classification des commandes (urgente, normale, différée)
  Option E : Prédiction du risque de rupture de stock

Pour le sujet choisi, rédige :

1.1 Contexte et problématique :
  - Contexte industriel/économique
  - Pourquoi ce problème est critique en supply chain
  - Formulation claire de la problématique sous forme de question

1.2 Objectifs du projet :
  - Objectif principal
  - 3 à 5 objectifs spécifiques
  - Résultats attendus

1.3 Revue de littérature :
  - Cite au moins 5 références académiques (articles, livres, conférences)
  - Présente les travaux antérieurs sur le sujet
  - Identifie les gaps que ton projet comble
  - Références au format APA

============================================================
PARTIE 4 : CHAPITRE 2 - CONTRACTUALISATION (2-3 pages)
============================================================

2.1 Cadre institutionnel :
  - Université Hassan 1er, ENCG Settat
  - Filière : Purchasing and Supply Chain Management
  - Semestre : S8
  - Module concerné : [préciser]
  - Ce projet s'inscrit dans la formation pratique visant à doter les étudiants
    de compétences en data-driven decision making appliqué aux achats
    et à la logistique

2.2 Périmètre et délimitation :
  - Ce que le projet couvre
  - Ce que le projet ne couvre PAS
  - Hypothèses de travail
  - Contraintes identifiées

2.3 Méthodologie :
  - Méthodologie CRISP-DM (Cross-Industry Standard Process for Data Mining)
    OU KDD (Knowledge Discovery in Databases)
  - Détaille chaque étape de la méthodologie choisie
  - Justifie le choix

2.4 Outils et technologies :
  - Python 3.x
  - Bibliothèques : pandas, numpy, scikit-learn, matplotlib, seaborn,
    plotly (si utilisé)
  - Environnement : Jupyter Notebook / Google Colab
  - Présente chaque outil avec une courte description

2.5 Planning :
  - Présente un diagramme de Gantt simplifié (sous forme de tableau)
  - Répartition des tâches si travail en groupe

============================================================
PARTIE 5 : CHAPITRE 3 - SCRIPT PYTHON COMPLET (5-7 pages)
============================================================

IMPORTANT : Le code doit être COMPLET, EXÉCUTABLE et utiliser des
DONNÉES RÉELLES ou un dataset public reconnu.

Datasets suggérés (choisis le plus adapté au sujet) :
  - "DataCo Global Supply Chain Dataset" (Kaggle)
  - "Supply Chain Shipment Pricing Dataset" (Kaggle)
  - "Brazilian E-Commerce Public Dataset by Olist" (Kaggle)
  - OU génère un dataset réaliste avec des caractéristiques crédibles

Structure du code Python :

```python
# ============================================================
# PROJET : [Titre du projet]
# ENCG Settat - S8 - Purchasing & Supply Chain Management
# Auteur(s) : [Noms]
# Date : [Date]
# ============================================================

# --- SECTION 1 : Importation des bibliothèques ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier  # ou autre modèle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# --- SECTION 2 : Chargement des données ---
# [Charger le dataset avec pd.read_csv ou le générer]
# Afficher les premières lignes, shape, info, describe

# --- SECTION 3 : Prétraitement ---
# Gestion des valeurs manquantes
# Encodage des variables catégorielles
# Feature engineering (création de nouvelles variables pertinentes)
# Normalisation/Standardisation
# Sélection des features

# --- SECTION 4 : Analyse Exploratoire (EDA) ---
# Distribution de la variable cible
# Corrélations (heatmap)
# Visualisations par catégorie
# Boxplots, histogrammes, countplots

# --- SECTION 5 : Modélisation ---
# Split train/test (80/20 ou 70/30)
# Entraînement de PLUSIEURS modèles :
#   - Régression Logistique
#   - Arbre de Décision
#   - Random Forest
#   - SVM (si applicable)
# Validation croisée (cross-validation)
# Optimisation des hyperparamètres (GridSearchCV si possible)

# --- SECTION 6 : Évaluation ---
# Classification report pour chaque modèle
# Matrice de confusion
# Courbe ROC et AUC
# Comparaison des modèles (tableau récapitulatif)

# --- SECTION 7 : Visualisations finales ---
# Matrice de confusion (heatmap)
# Courbe ROC
# Feature importance (bar chart)
# Comparaison des accuracies des modèles (bar chart)
```

IMPORTANT :
- Chaque section doit être précédée de commentaires explicatifs
- Le code doit pouvoir tourner de bout en bout
- Inclure des print() pour afficher les résultats intermédiaires
- Utiliser plt.savefig() pour sauvegarder les graphiques

============================================================
PARTIE 6 : CHAPITRE 4 - ANALYSE DES RÉSULTATS (3-4 pages)
============================================================

4.1 Métriques d'évaluation :
  - Présente les résultats sous forme de TABLEAU pour chaque modèle :
    | Modèle              | Accuracy | Precision | Recall | F1-Score | AUC  |
    |----------------------|----------|-----------|--------|----------|------|
    | Régression Logistique| xx%      | xx%       | xx%    | xx%      | x.xx |
    | Arbre de Décision    | xx%      | xx%       | xx%    | xx%      | x.xx |
    | Random Forest        | xx%      | xx%       | xx%    | xx%      | x.xx |
  - Les valeurs doivent être RÉALISTES et COHÉRENTES

4.2 Interprétation :
  - Quel modèle est le plus performant et POURQUOI
  - Quelles features sont les plus importantes et leur signification métier
  - Lien entre les résultats statistiques et la réalité supply chain

4.3 Comparaison :
  - Avantages et inconvénients de chaque modèle dans ce contexte
  - Quel modèle recommander pour une mise en production

4.4 Discussion et limites :
  - Limites du dataset
  - Limites des modèles
  - Biais potentiels
  - Pistes d'amélioration

============================================================
PARTIE 7 : CHAPITRE 5 - ILLUSTRATIONS (2-3 pages)
============================================================

Décris précisément chaque illustration qui sera générée par le code Python :

5.1 - Figure 1 : Distribution de la variable cible (bar chart / pie chart)
      → Description et interprétation

5.2 - Figure 2 : Heatmap de corrélation
      → Description et interprétation

5.3 - Figure 3 : Matrice de confusion du meilleur modèle
      → Description et interprétation

5.4 - Figure 4 : Courbe ROC comparative
      → Description et interprétation

5.5 - Figure 5 : Feature Importance (Top 10)
      → Description et interprétation

5.6 - Figure 6 : Comparaison des accuracies des modèles (barplot)
      → Description et interprétation

Pour chaque figure :
  - Donne un titre
  - Explique ce qu'elle montre
  - Interprète les résultats
  - Relie à la problématique supply chain

============================================================
PARTIE 8 : CONCLUSION GÉNÉRALE (1 page)
============================================================

- Rappel de la problématique
- Synthèse des résultats obtenus
- Apports du projet (académiques et professionnels)
- Recommandations pour les professionnels de la supply chain
- Perspectives et travaux futurs
- Ouverture (Deep Learning, données temps réel, IoT...)

============================================================
PARTIE 9 : BIBLIOGRAPHIE (format APA)
============================================================

Liste au moins 10 références :
- 5 articles académiques minimum
- 2 livres minimum
- 3 ressources web (Kaggle, documentation scikit-learn, etc.)

============================================================
PARTIE 10 : ANNEXES
============================================================

- Code Python complet (si non intégré dans le corps du rapport)
- Captures d'écran de l'exécution
- Dictionnaire des données (description de chaque variable du dataset)
- Tout élément complémentaire

============================================================
CONSIGNES DE FORMATAGE
============================================================

- Utiliser une numérotation cohérente (1, 1.1, 1.1.1...)
- Chaque chapitre commence sur une nouvelle page
- Les figures sont numérotées : Figure 1, Figure 2...
- Les tableaux sont numérotés : Tableau 1, Tableau 2...
- Police suggérée : Times New Roman 12pt, interligne 1.5
- Marges : 2.5 cm de chaque côté
- Pagination en bas de page centrée
- En-tête : "ENCG Settat - S8 - PSCM" à droite
- Pied de page : numéro de page

============================================================
CONTRAINTES OBLIGATOIRES
============================================================

1. Le compte rendu doit faire entre 25 et 40 pages
2. Le code Python doit être COMPLET et FONCTIONNEL
3. Les données doivent être RÉELLES (dataset public) ou générées
   de manière RÉALISTE avec np.random.seed() pour reproductibilité
4. Les résultats doivent être COHÉRENTS et RÉALISTES
5. Le ton doit être ACADÉMIQUE et PROFESSIONNEL
6. Toutes les figures doivent avoir un titre, des axes labellisés,
   et une légende si nécessaire
7. Le lien avec la filière Purchasing & Supply Chain Management
   doit être EXPLICITE tout au long du rapport

============================================================

GÉNÈRE MAINTENANT LE COMPTE RENDU COMPLET EN SUIVANT
EXACTEMENT CETTE STRUCTURE.
```

---

## 🎯 COMMENT UTILISER CE PROMPT

### Étape 1 — Personnalisation
Avant de soumettre le prompt, remplacez les éléments entre crochets :
| Placeholder | Remplacer par |
|---|---|
| `[Nom(s) de(s) l'étudiant(s)]` | Vos noms complets |
| `[Nom du professeur]` | Le nom de votre encadrant |
| `[Date]` | La date de soumission |
| `Option A/B/C/D/E` | Précisez le sujet voulu ou laissez l'IA choisir |

### Étape 2 — Soumission
Soumettez le prompt en **une seule fois** dans un modèle capable de réponses longues (Claude, GPT-4, etc.).

### Étape 3 — Si la réponse est coupée
Relancez avec :
```
Continue exactement là où tu t'es arrêté. 
Reprends depuis la dernière section générée.
```

### Étape 4 — Extraction du code Python
Demandez séparément :
```
Donne-moi UNIQUEMENT le script Python complet du Chapitre 3 
dans un seul bloc de code, prêt à être exécuté dans Google Colab, 
avec toutes les installations pip nécessaires en première ligne.
```

### Étape 5 — Génération des figures
Exécutez le script Python dans **Google Colab** ou **Jupyter Notebook** pour générer les graphiques réels, puis intégrez-les dans votre rapport Word/LaTeX.

---

## 📌 VARIANTES OPTIONNELLES

Si vous voulez un **sujet de prédiction** plutôt que de classification, ajoutez au prompt :
```
MODIFICATION : Utilise un modèle de RÉGRESSION (prédiction d'une variable 
continue) au lieu de classification. 
Adapte les métriques : utilise RMSE, MAE, R² au lieu de accuracy/precision/recall.
Adapte les visualisations : scatter plot réel vs prédit, résidus, etc.
```

Si vous voulez le rapport en **LaTeX** :
```
MODIFICATION : Génère le compte rendu complet en format LaTeX (.tex), 
compilable directement, avec les packages nécessaires 
(geometry, graphicx, listings, hyperref, biblatex...).
```

---

Ce prompt est conçu pour produire un livrable **complet, professionnel et directement utilisable** pour votre module S8 à l'ENCG Settat.
