# les bibliothèques de bases
import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib import pyplot as plt
import plotly.graph_objects as go

import streamlit as st
import io
import gzip
# mise a l'échelle des données
from sklearn.preprocessing import StandardScaler
# Separation train/test data
from sklearn.model_selection import train_test_split

# données déséquilibrés

from imblearn.over_sampling import SMOTE # RandomOverSampler

# les biblio liées directement aux models
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Metric de Performance 
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,recall_score,precision_score,classification_report,RocCurveDisplay,f1_score,precision_recall_curve,roc_auc_score,PrecisionRecallDisplay

# sauvegarde et chargement du modele
from joblib import dump,load


def main():

    # je mets l'entete de l'application
    # st.title("web apps de machine learning pour la détection de fraude par carte de crédit bancaire")
    # Titre de l'application en gras, rouge et sur toute la largeur de la page
    import streamlit as st

    # Ajouter un message défilant
    st.markdown(
        """
        <style>
        .marquee {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            animation: marquee 10s linear infinite;
            font-weight: bold;  # Mettre le texte en gras
        }
        @keyframes marquee {
            0% { transform: translate(100%, 0); }
            100% { transform: translate(-100%, 0); }
        }
        </style>
        <div class="marquee">
            <p>🚀 Découvrez comment notre modèle de détection de fraude révolutionne la sécurité des transactions ! 🌟</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
    """
    <h1 style='text-align: center; color: #FFA500; font-weight: bold;'>
        👮🏻 Bienvenue dans le système intelligent de détection de fraude par carte de crédit 👋👮🏻‍♀️
    </h1>
    """,
    unsafe_allow_html=True
    )

    # sous titres (ici je veux mettre'auteur)
    st.subheader(' Auteur : Didier Matton, Data Scientist-Analyst')
    # sous titres (l'objectif)
    st.subheader('Arrière Plan ')
    st.write("Solution pilotée par l’IA pour réduire la fraude par carte de paiement. L’objectif est d’utiliser des techniques avancées d’apprentissage automatique pour prédire avec précision la probabilité de fraude par carte de crédit, ce qui permet d'effectuer les décisions de paiement en toute sécurité.")

    # j'importe les données
    @st.cache_data(persist=True)
 
    def load_data():
        with gzip.open('creditcard.csv.gz', 'rt', encoding='utf-8') as f:
            data = pd.read_csv(f)
        return data.copy()

    # affichage d'un échantillon du dataframe
    df = load_data()

    @st.cache_data(persist=True)
    def getDataframeInfo(df:pd.DataFrame)->pd.DataFrame:
        buf = io.StringIO()
        df.info(buf=buf)
        lines = buf.getvalue().splitlines()
        return (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
        .drop('Count',axis=1)
        .rename(columns={'Non-Null':'Non-Null Count'}))
    # affichage des information du dataframe
    info = getDataframeInfo(df).reset_index().drop(['index','#'],axis=1)
    null= pd.DataFrame(df.isnull().sum().sort_values(ascending=False)).transpose()
    num_cols = len(df.select_dtypes(include=np.number).drop("Class", axis=1).columns)
    features = len(df.drop("Class",axis=1).columns)
    

    @st.cache_data(persist=True)
    def get_target_distribution(dataset:pd.DataFrame):
        temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                            height=500, width=1000))
        target=dataset["Class"].value_counts(normalize=True)
        target.rename(index={0:'Transaction Authentique',1:'Transaction frauduleuse' },inplace=True)
        pal, color=['#016CC9','#DEB078'], ['#8DBAE2','#EDD3B3']
        fig=go.Figure()
        fig.add_trace(go.Pie(labels=target.index, values=target*100, hole=.45, 
                            showlegend=True,sort=False, 
                            marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                            hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
        fig.update_layout(template=temp, title='Target Distribution', 
                        legend=dict(traceorder='reversed',y=1.05,x=0),
                        uniformtext_minsize=15, uniformtext_mode='hide',width=700)
        return fig
    @st.cache_data(persist=True)
    def get_numeric_distribution(data:pd.DataFrame, target_column:str):
        names = data.columns.drop(target_column)
        figs =[]
        fig1, axes = plt.subplots(1,2, squeeze=False)
        sns.boxplot(y=names[0], x= target_column, data=data, orient='v', ax=axes[0,0], hue='Class')
        sns.boxplot(y=names[1], x= target_column, data=data, orient='v', ax=axes[0,1], hue='Class')
        figs.append(fig1)

        fig2, axes2 = plt.subplots(1,2, squeeze=False)
        sns.boxplot(y=names[2], x= target_column, data=data, orient='v', ax=axes2[0,0], hue='Class')
        sns.boxplot(y=names[3], x= target_column, data=data, orient='v', ax=axes2[0,1], hue='Class')
        figs.append(fig2)

        fig3, axes = plt.subplots(1,2, squeeze=False)
        sns.boxplot(y=names[4], x= target_column, data=data, orient='v', ax=axes[0,0], hue='Class')
        sns.boxplot(y=names[5], x= target_column, data=data, orient='v', ax=axes[0,1], hue='Class')
        figs.append(fig3)
        
        fig4, axes = plt.subplots(1,1, squeeze=False)
        sns.boxplot(y=names[6], x= target_column, data=data, orient='v', ax=axes[0,0], hue='Class')
        figs.append(fig4)
        return figs
    
    @st.cache_data(persist=True)
    def wrangle(dataset:pd.DataFrame):
        # Calcul des  corrélations avec la variable cible 
        correlations = dataset.corr()['Class'].abs()
        # Sélection des caractéristiques avec une corrélation absolue supérieure à 0,09
        selected_features = correlations[correlations > 0.09].index
        # Création d'un nouveau DataFrame avec les caractéristiques sélectionnées
        df_selected= dataset[selected_features]
        return df_selected.copy()

    df1=wrangle(df)
    Seed = 101
    # 
    @st.cache_data(persist=True)
    def split(data : pd.DataFrame):
        y=data["Class"]
        X=data.drop('Class',axis=1)
        return train_test_split(X,y,test_size=0.2,random_state=Seed,stratify=y)
      # Appel de la fonction split
    X_train, X_test, y_train, y_test = split(df1)
    
    @st.cache_data(persist=True)
    def scale_data():
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(X_train)
        Scaled_test = scaler.transform(X_test)
        return scaled_train.copy(),Scaled_test.copy()
    scaled_train,Scaled_test = scale_data()

    @st.cache_data(persist=True)
    def smote_data ():
        SMOTE_sampler = SMOTE(sampling_strategy='minority',random_state=Seed)
        return SMOTE_sampler.fit_resample(scaled_train,y_train)

    X_train_over, y_train_over = smote_data()
        
      # Analyse de la performance des modeles
    @st.cache_data(persist=True)
    def plot_perf(graphes,y_true,y_pred):
        if "Matrice de Confusion" in graphes :
            st.subheader('Matrice de Confusion')
            fig, ax = plt.subplots(dpi=200)
            ConfusionMatrixDisplay.from_predictions(y_true,y_pred,ax=ax,display_labels=class_names)
            st.pyplot(fig)

        if "ROC curve" in graphes :
            st.subheader('ROC curve')
            fig, ax = plt.subplots(dpi=200)
            RocCurveDisplay.from_predictions(y_true,y_pred,ax=ax)
            st.pyplot(fig)

        
        if "Precision-Recall curve" in graphes :
            st.subheader('Precision-Recall curve')
            fig, ax = plt.subplots(dpi=200)
            PrecisionRecallDisplay.from_predictions(y_true,y_pred,ax=ax)
            st.pyplot(fig)

            # recuperer les informations
    @st.cache_data(persist=True)
    def get_params_model(_model, acc_train, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score):
        model_perfs = pd.DataFrame({
            'Model': [str(_model)], 
            'Accuracy_train' : [acc_train],
            'Accuracy': [accuracy_score], 
            'Precision': [precision_score],
            'Recall': [recall_score],
            'F1': [f1_score],
            'ROC-AUC': [roc_auc_score]
        }).sort_values('Accuracy', ascending=False).reset_index(drop=True)
        return model_perfs.transpose()


    if st.sidebar.checkbox("Analyse Exploratoire des Données : jeu de données crédit card",False):
        st.subheader('Analyse Exploratoire des Données')
        st.dataframe(df.head(20))
        st.markdown(f"**Taille du jeu de données :** {len(df)}")
        st.dataframe(info)
        st.write('Valeur nulle')
        st.dataframe(null)
        st.markdown(f"**Caractéristiques ou features :** {features}")
        st.markdown(f"**Variable Cible ou Target :** {1}")
        st.markdown(f"**Colonnes numériques :** {num_cols}")
        st.markdown(f"**Colonnes Catégorielles :** {features- num_cols}")
        # je calcul les statistiques descriptive de l'échantillon
        st.subheader('Distribution des données ')
        st.dataframe(df.describe().transpose())
        # Matrice de correlation
        st.subheader('Matrice de correlation ')
        st.dataframe(df.corr())
        st.markdown("""
            A ce stade, on constate que :\n\
                - il n'existe pas de multicolinéarité entre les caractérisriques;\
                Mais il y'a des variables qui ont une corrélation très peu importante par rapport à la variable cible\n
                - on va retirer ses variables dans le but d'améliorer nos estimations. 
                - notre règle de décision sera de retirer toute variable donc la valeur obsolue est inférieure à 0.09
            """)
    # Graphiques de l'analyse exploratoire

        st.plotly_chart(get_target_distribution(df))
        st.subheader("Analyse d'outliers (o données aberrantes)" )
        figs = get_numeric_distribution(data=df, target_column='Class')
        for fig in figs:
            st.pyplot(fig)

    if st.sidebar.checkbox("Ingénierie des fonctionnalités",False):
        st.subheader('Ingénirie des fonctionnalités')
        st.markdown("""
        Ici :
        - Nous allons travailler avec des variables qui ont une corrélation supérieure à 0.09 avec la variable cible;
        - Convertir les variables catégorielles si elles existaient;
        - Supprimer les doublons et autres...
        """)
        st.write('Variables restantes et corrélation avec la target ')
        # Création de la heatmap
        fig, ax = plt.subplots(dpi=200,figsize=(16,8))
        sns.heatmap(data=pd.DataFrame(df1.corr()), annot=True, ax=ax)

        # Affichage de la heatmap dans Streamlit
        st.pyplot(fig)

    if st.sidebar.checkbox("Selection de l'achictecture du modèle",False):
        st.header("Model Achitecture Selection")
        st.write("**Pour le modèle logistique régression avec suréchantillonage SMOTE**")
        st.markdown("""
        - **Train X**  : (454902, 14) ; 
        - **Train y** : (454902,) ;
        - **Test X**  : (56962, 14) ; 
        - **Test y** : (56962,) ;
                    
        """)

        st.write("**Pour le modèle Balanced Random Forest (class_weight = 'balanced')**")
        st.markdown("""
        - **Train X**  : (227845, 14) ; 
        - **Train y** : (227845,) ;
        - **Test X**  : (56962, 14) ; 
        - **Test y** : (56962,) ;
                    
        """)

        st.write("**Pour le modèle XGBoost (scale_pos_weight = 577)**")
        st.markdown("""
        - **Train X**  : (227845, 14) ; 
        - **Train y** : (227845,) ;
        - **Test X**  : (56962, 14) ; 
        - **Test y** : (56962,) ;
                    
        """)
    
        
        st.markdown("""
        - Sur les 3 architectures, seule LGBMClassifier n’est pas en surajustement; 
        - Le but sera donc d’améliorer les hyperparamètres pour avoir un meilleur LGBMClassifier" ;
        """)
        # je Crée un DataFrame avec les métriques de performance des modèles
        result = {
        'Modèle': ['LogisticRegression()', 'BalancedRandomForestClassifier()', 'XGBClassifier()'],
        'acc_train': [0.93242, 0.97065, 0.94525],
        'accuracy_score': [0.99919, 0.99946, 0.99958],
        'precision_score': [0.74074, 0.8764, 0.87755],
        'recall_score': [0.81633, 0.79592, 0.87755],
        'f1_score': [0.7767, 0.83422, 0.87755],
        'roc_auc_score': [0.90792, 0.89786, 0.93867]
         }
        st.table(pd.DataFrame(result).transpose())

        # Conclusion synthétique
        conclusion = """
        ### Conclusion Synthétique

        À la lumière des métriques les plus pertinentes pour notre datasets, **XGBoost** se distingue comme le meilleur modèle pour plusieurs raisons :

        1. **Précision et Rappel**: XGBoost affiche des scores de précision et de rappel de 87.75%, ce qui signifie qu'il identifie correctement une grande proportion de fraudes tout en minimisant les faux positifs. Cet équilibre est crucial pour la détection de fraude où les deux types d'erreurs peuvent avoir des conséquences significatives.

        2. **F1 Score**: Avec un F1 score de 87.75%, XGBoost montre qu'il maintient un excellent équilibre entre précision et rappel, ce qui est essentiel pour des données déséquilibrées comme celles de la fraude par carte de crédit.

        3. **ROC AUC Score**: Le score ROC AUC de 0.93867 indique une excellente capacité de discrimination entre les transactions frauduleuses et non frauduleuses. Cela montre que XGBoost est très efficace pour différencier les deux classes.

        4. **Performance Générale**: Bien que tous les modèles aient des performances élevées, XGBoost surpasse légèrement les autres en termes de précision, rappel, et ROC AUC score, ce qui en fait le choix optimal pour notre application.
        """

        # Affichage de la conclusion
        st.markdown(conclusion)

        

    
    class_names=['T. Authentique','T.Frauduleuse']

    st.sidebar.write(f"**Simulation de l'achictecture du Modèle**")
    classifier=st.sidebar.selectbox(
        "Classificateur",
        ('Logistic Regression avec Suréchantillonage SMOTE','Balanced Random Forest',"XGBoost")
         )
    

    # Logistic Regression
    if classifier=="Logistic Regression avec Suréchantillonage SMOTE":
         # reglage des hyperparametres du model
        st.sidebar.subheader('Hyperparamètres du modèle')
        C=st.sidebar.number_input("choisir la valeur du parametres de régularisation **'C'** (Inverse de la force de régularisation )",
            min_value=0.01,max_value=15.0)
        max_iter=st.sidebar.number_input("choisir le nombre maximum d'itération **'max_iter'**",
                                         min_value=100,max_value=1000,step=10)
        penalty =st.sidebar.radio("Précisez la norme de la pénalité **'penalty'** ",('l1','l2','elasticnet'))
        l1_ratio = st.sidebar.number_input("choisir le paramètre de contrôle le mélange des pénalités L1 et L2 dans ElasticNet **'l1_ratio'** ",
                                           min_value=np.linspace(0,1,20).min(),max_value=np.linspace(0,1,20).max())
       
        #  je mets mes graphiques de performances 
        graphes_perf=st.sidebar.multiselect(
            "Choisir graphique (s) de performance du modèle ML",
            ("Matrice de Confusion","ROC curve", "Precision-Recall curve"))

        # pour le modele
       

        if st.sidebar.button('Execution de la simulation',key='Classify'):
            st.subheader('Résultat Simulation Logistique Regression')
        
        # j'instancie mon modèle
            model=LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=Seed,
                penalty=penalty,
                l1_ratio = l1_ratio,
                solver='saga',
            )

        # J'entraîne le modèle
            model.fit(X_train_over, y_train_over)
        # Affichage des résultats (à compléter selon tes besoins)
            st.write("Modèle entraîné avec succès !")
        
        # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)
        # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
        
        # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))
        
        # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf,y_true=y_test,y_pred=y_pred_proba)  

        st.sidebar.write(f"**Model_achictecture_selection : Best Model**")
        if st.sidebar.button('Execution du Best model', key='Best_logistique'):
            st.subheader('Résultat Meilleur Estimateur Logistique Regression ')
            
            # j'instancie mon modèle
            model = LogisticRegression(
                C=23357214.690901212,
                max_iter=120,
                random_state=Seed,
                penalty='elasticnet',
                l1_ratio=0.3684210526315789,
                solver='saga',
            )

            # J'entraîne le modèle
            model.fit(X_train_over, y_train_over)
            st.write("Modèle entraîné avec succès !")
            
            # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)
            # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
            # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))
            
            # je récuperes les metrics et le  model
            df_log = get_params_model(model,acc_train, accuracy, precision, recall, f1, auc)
            st.dataframe(df_log)

            # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf, y_true=y_test, y_pred=y_pred_proba)

           


      # Balanced Random Forest
    if classifier=="Balanced Random Forest":
         # reglage des hyperparametres du model
        st.sidebar.subheader('Hyperparamètres du modèle')
        n_estimators=st.sidebar.number_input("choisir le nombre d'arbres dans la forêt **'n_estimators'** ",
            min_value=64,max_value=500)
        max_features=st.sidebar.number_input("choisir le nombre de fonctionnalités à prendre en compte lors de la recherche de la meilleure répartition **'max_features'**",
            min_value=2,max_value=4,step=1)
        bootstrap =st.sidebar.radio("Précisez si des échantillons bootstrap sont utilisés lors de la construction des arbres **'bootstrap'** ",('True','False'))
        # Conversion de la sélection en booléen
        bootstrap = True if bootstrap == "True" else False
        oob_score =st.sidebar.radio("Précisez s'il faut utiliser des échantillons prêts à l’emploi pour estimer la précision de la généralisation **'oob_score'** ",('True','False'))
        # Conversion de la sélection en booléen
        oob_score = True if oob_score == "True" else False
        criterion =st.sidebar.radio("Précisez la fonction permettant de mesurer la qualité d'une division **'criterion'** ",('gini','entropy'))


       
        #  je mets mes graphiques de performances  , 
        graphes_perf=st.sidebar.multiselect(
            "Choisir graphique (s) de performance du modèle ML",
            ("Matrice de Confusion","ROC curve", "Precision-Recayll curve"))

        # pour le modele
       

        if st.sidebar.button('Execution de la simulation',key='Classify'):
            st.subheader('Résultat Simulation Balance Random Forest')
        
        # j'instancie mon modèle
            model=BalancedRandomForestClassifier(
                n_estimators = n_estimators,    
                max_features = max_features,
                random_state = Seed,
                bootstrap = bootstrap,
                oob_score = oob_score,
                criterion=criterion,
                class_weight='balanced',
                n_jobs= -1
            )

        # J'entraîne le modèle
            model.fit(scaled_train, y_train)
        # Affichage des résultats (à compléter selon tes besoins)
            st.write("Modèle entraîné avec succès !")
        
         # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)

            # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
            
        
        # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))
        
        # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf,y_true=y_test,y_pred=y_pred_proba)  

        st.sidebar.write(f"**Model_achictecture_selection : Best Model**")
        if st.sidebar.button('Execution du Best model', key='Best_random_model'):
            st.subheader('Résultat Meilleur Estimateur Balanced Random Forest')
            
            # j'instancie mon modèle
            model=BalancedRandomForestClassifier(
                n_estimators = 128,    
                max_features = 3,
                random_state = Seed,
                bootstrap = True,
                oob_score = True,
                criterion='gini',
                class_weight='balanced',
                n_jobs= -1
            )

            # J'entraîne le modèle
            model.fit(scaled_train, y_train)
            st.write("Modèle entraîné avec succès !")
            
            # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)
            
            # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
            # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))

            # je récuperes les metrics et le  model
            df_random = get_params_model(model,acc_train, accuracy, precision, recall, f1, auc)
            st.dataframe(df_random)
            
            # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf, y_true=y_test, y_pred=y_pred_proba)
    
    

      # XGBoost
    if classifier=="XGBoost":
         # reglage des hyperparametres du model
        st.sidebar.subheader('Hyperparamètres du modèle')
        n_estimators = st.sidebar.number_input("choisir le nombre d'arbres dans le modèle **'n_estimators'** ",
            min_value=100,max_value=500)
        max_depth = st.sidebar.number_input("Choisir la profondeur maximale des arbres. Contrôle la complexité du modèle **'max_depht'**",
            min_value=3,max_value=8,step=1)
        learning_rate = st.sidebar.number_input("Contrôler la taille du pas à chaque itération pendant que le modèle s’optimise vers son objectif **'learning_rate'**",
            min_value=0.01,max_value=0.2,step=0.05)
        subsample = st.sidebar.number_input(" Préciser la fraction des échantillons à utiliser pour chaque arbrf **'subsample'**",
            min_value=0.6,max_value=1.0,step=0.1)
        colsample_bytree = st.sidebar.number_input(" Préciser la fraction des caractéristiques à considérer pour chaque arbre.**'colsample_bytree'**",
            min_value=0.6,max_value=1.0,step=0.1)

        # Calculer le ratio de déséquilibre
        ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
       
        #  je mets mes graphiques de performances  , 
        graphes_perf=st.sidebar.multiselect(
            "Choisir graphique (s) de performance du modèle ML",
            ("Matrice de Confusion","ROC curve", "Precision-Recayll curve"))

        # pour le modele
       

        if st.sidebar.button('Execution de la simulation',key='Classify'):
            st.subheader('Résultat Simulation XGBoost')
        
        # j'instancie mon modèle
            model = XGBClassifier(
                n_estimators = n_estimators,    
                max_depth = max_depth ,
                random_state = Seed,
                learning_rate = learning_rate,
                subsample = subsample,
                colsample_bytree = colsample_bytree,
                scale_pos_weight=ratio,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method="hist", 
                early_stopping_rounds=2,
                verbose = 0,
                n_jobs= -1
            )
        # je met scaled_train au standard de x_test
            scaled_train = pd.DataFrame(scaled_train,columns=X_test.columns)
        # j'entraîne le modèle avec un ensemble de validation
            model.fit(scaled_train ,y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Affichage des résultats (à compléter selon tes besoins)
            st.write("Modèle entraîné avec succès !")
        
         # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)

            # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
            
        
        # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))
        
        # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf,y_true=y_test,y_pred=y_pred_proba)  

        st.sidebar.write(f"**Model_achictecture_selection : Best Model**")
        if st.sidebar.button('Execution du Best model', key='Best_xgb_model'):
            st.subheader('Résultat Meilleur Estimateur XGBoost')
            
           # j'instancie mon modèle
            model = XGBClassifier(
                n_estimators = 500,    
                max_depth = 8 ,
                random_state = Seed,
                learning_rate = 0.15,
                subsample = 1.0,
                colsample_bytree = 0.6,
                scale_pos_weight=ratio,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method="hist", 
                early_stopping_rounds=2,
                verbose = 0,
                n_jobs= -1
            )
        # je met scaled_train au standard de x_test
            scaled_train = pd.DataFrame(scaled_train,columns=X_test.columns)
        # j'entraîne le modèle avec un ensemble de validation
            model.fit(scaled_train ,y_train, eval_set=[(X_test, y_test)], verbose=True)
            
            # Prédictions
            y_pred = pd.DataFrame(model.predict(Scaled_test))
            y_pred_proba = pd.DataFrame(model.predict_proba(Scaled_test)[:, 1])
            
            # Calcul des courbes de précision-rappel
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            # calcul du le seuil optimal
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            # Ajustement du seuil de décision
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            y_pred_proba = (y_pred_proba >= optimal_threshold).astype(int)
            
            # métric de performance
            f1 = np.round(f1_score(y_test, y_pred_proba), 5)
            auc = np.round(roc_auc_score(y_test, y_pred_proba), 5)
            acc_train = accuracy_score(y_train_over,model.predict(X_train_over))
            accuracy = np.round(accuracy_score(y_test, y_pred_proba), 5)
            precision = np.round(precision_score(y_test, y_pred_proba), 5)
            recall = np.round(recall_score(y_test, y_pred_proba), 5)
            
            # afficher les metric dans l'application
            st.markdown('**F1-Score :** ' + str(f1))
            st.markdown('**AUC-ROC (Area Under the Receiver Operating Characteristic Curve) :** ' + str(auc))
            st.markdown('**Accuracy _train:** ' + str(acc_train))
            st.markdown('**Accuracy :** ' + str(accuracy))
            st.markdown('**Precision :** ' + str(precision))
            st.markdown('**Recall :** ' + str(recall))
            
            # je récuperes les metrics et le  model
            df_xgb = get_params_model(model,acc_train, accuracy, precision, recall, f1, auc)
            st.dataframe(df_xgb)

            # afficher les graphiques de performances
            plot_perf(graphes=graphes_perf, y_true=y_test, y_pred=y_pred_proba)

            

    
        
          


if __name__ == '__main__':
    main()
