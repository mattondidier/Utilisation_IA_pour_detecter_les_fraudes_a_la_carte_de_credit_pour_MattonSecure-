
```markdown
# Détection de Fraude par Carte de Crédit avec l'IA pour MattonSecure

## Contexte

En tant qu'Officier ML chez MattonSecure, la tâche consiste à développer une solution basée sur l'IA pour détecter les fraudes par carte de crédit. L'objectif est d'utiliser des techniques avancées de machine learning pour prédire avec précision les transactions frauduleuses, aidant ainsi à prendre des décisions financières éclairées.

## Objectif

Construire un modèle ML robuste prédisant la probabilité de fraude par carte de crédit, en intégrant diverses sources de données pour améliorer les processus de détection et réduire les taux de fraude.

### Base de Données :

#### Sources de Données :

1. **Données de Transactions de Carte de Crédit :** Informations sur les transactions, montants, le temps, résultat PCA sur les variables etc.
2. **Informations Bancaires :** Soldes des comptes, historique des transactions, fréquence des découverts (via les API bancaires ouvertes).
3. **jeu de données** : jeu de données de détection de fraude par carte de crédit de kaggle. 



## Résolution Conceptuelle et Analytique des Problèmes

### Développement des Caractéristiques pour l'Évaluation des fraudes

Identifier les caractéristiques pour évaluer la fraude par carte de  crédit, en tenant compte des diverses sources de données pour mieux apprender le concept.

### Sélection du Modèle pour la Version 1

Décider du modèle le mieux adapté pour le déploiement initial. Partager le processus de prise de décision.

### Analyse Post-Déploiement de la Version 1 (77% score F1)

Identifier les domaines d'amélioration par des analyses détaillées.

### Identification des Biais du Modèle

Déterminer et identifier les biais contre certaines transactions.

### Prochaine Version du Modèle (87.5% score F1)

après une recherche par grille afin d'optimiser la metrique principale adaptée au données déséquilibrés,considérations et actions avant de déployer le modèle.

### Stratégie de Déploiement

Comment intégrer le modèle dans le flux de travail existant.

### Mécanismes de Surveillance

Établir une surveillance des performances du modèle et des dérives de données.

### Prochaines Étapes

Mettre en place des protocoles et processus supplémentaires pour une amélioration continue.

## Livrables (Délai de 7 Jours)

- Un modèle prédictif identifiant les transactions frauduleuses.
- Une stratégie de déploiement avec des protocoles de surveillance et de mise à jour.
- une application web permetant de présenter le travail et faire des simulation sur le modèle retenu. 

## Tâches

### Intégration des Données et Conformité à la Protection des Données

Assurer la conformité aux lois sur la protection des données (par exemple, RGPD, CCPA, Loi n° 2010/012 du 21 décembre 2010 au cameroun, Protection of Personal Information Act (POPIA)... ) et intégrer les différentes sources de données en un ensemble de données unifié.

### Analyse Exploratoire des Données (EDA) et Prétraitement

Réaliser une EDA pour comprendre les distributions de données, les corrélations et les motifs. Nettoyer et prétraiter les données, en traitant les valeurs manquantes, les valeurs aberrantes et l'encodage des caractéristiques.

### Ingénierie des Caractéristiques

Développer des caractéristiques capturant les déterminant de la fraude par carte de crédit. Utiliser les connaissances du domaine pour la construction de caractéristiques prédictives.

### Développement et Sélection du Modèle

Expérimenter avec divers modèles ML et effectuer un réglage des hyperparamètres pour l'optimisation.

### Évaluation du Modèle

Utiliser des métriques comme l'AUC-ROC, la précision, le F1-score (critère de décision lors de l'évaluation des modèles) et la précision-rappel. Se concentrer sur la minimisation des faux négatifs impactant les décisions financières.

### Évaluation des Biais et de l'Équité

Évaluer et atténuer les biais pour garantir des pratiques de paiement équitables.

### Déploiement et Surveillance

Développer une stratégie de déploiement et établir une surveillance des performances et des dérives de données.

### Boucle de Rétroaction et Amélioration Continue

Mettre en place un système pour recueillir les avis des utilisateurs de carte de crédit. Mettre régulièrement à jour le modèle avec de nouvelles données.

## Livrables

- Un rapport complet détaillant l'approche, les décisions architecturales, le processus de développement, les insights du modèle et l'évaluation de l'équité.
- Un modèle prédictif pour identifier les transactions frauduleuses.
- Une stratégie de déploiement avec des protocoles de surveillance et de mise à jour.

## Résultats des Modèles

### Modèles Utilisés et Résultats

- **Modèles** : Logistic Regression, Balanced Random Forest, XGBoost
- **Résultats** :
    - **Logistic Regression** :
        - Accuracy (train) : 0.93242
        - Accuracy (test) : 0.99919
        - Precision : 0.74074
        - Recall : 0.81633
        - F1 Score : 0.7767
        - ROC AUC Score : 0.90792
    - **Balanced Random Forest** :
        - Accuracy (train) : 0.97065
        - Accuracy (test) : 0.99946
        - Precision : 0.8764
        - Recall : 0.79592
        - F1 Score : 0.83422
        - ROC AUC Score : 0.89786
    - **XGBoost** :
        - Accuracy (train) : 0.94525
        - Accuracy (test) : 0.99958
        - Precision : 0.87755
        - Recall : 0.87755
        - F1 Score : 0.87755
        - ROC AUC Score : 0.93867

### Application Streamlit

Une application Streamlit a été conçue pour visualiser les données, présenter l'ingénierie des fonctionnalités, effectuer des simulations avec différents modèles et présenter les meilleurs modèles.

>>>>>>> edf65b3 (first commit)
"# Utilisation_IA_pour_detecter_les_fraudes_a_la_carte_de_credit_pour_MattonSecure" 
