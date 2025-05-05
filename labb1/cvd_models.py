import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score


# ------------------------------- 
# Data Loading for EDA
# -------------------------------
class DataLoader:                                           
    """Skapar en klass för att läsa in data från en CSV-fil.""" 
    def __init__(self, filepath, sep=";"):
        self.filepath = filepath
        self.sep = sep
        self.df = None

    def load_data(self):
        """Läser in data från en CSV-fil med angiven separator."""
        self.df = pd.read_csv(self.filepath, sep=self.sep)
        return self.df.copy()
#Kodhjälp: ChatGPT, prompt: "Hur skapar jag en klass i en py-fil för att läsa in data från en CSV-fil?"


# -------------------------------
# Data Cleaning, Processing & Feature Engineering for EDA # Kodhjälp: ChatGPT, prompt: "Min ipynb-fil "LaborationML" har för mycket kod. Kan du ge mig ett exempel på hur jag kan sukturera den i klasser baserat på de titlar jag har i ipynb-filen?"
# -------------------------------
class DataProcessor:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def initial_data_process(self):
        """ Databehnadling för tydligare visualisering"""
        
        # Ta bort kolumner som inte behövs
        self.df = self.df.drop(columns=["id"], axis=1)
        # Skapar nya kolumner och ge dem etiketter
        self.df["cholesterol_label"] = self.df["cholesterol"].map({1: "Normal", 2: "Above Normal", 3: "Way Above Normal"})
        self.df["gluc_label"] = self.df["gluc"].map({1: "Normal", 2: "Above Normal", 3: "Way Above Normal"})
        self.df["smoke_label"] = self.df["smoke"].map({0: "No", 1: "Yes"})
        self.df["alco_label"] = self.df["alco"].map({0: "No", 1: "Yes"})
        self.df["gender_label"] = self.df["gender"].map({1: "Female", 2: "Male"})
        self.df["CVD"] = self.df["cardio"].map({1: "Yes", 0: "No"})
         # Ändra värden i kolumn för enklare läsning
        self.df["age"] = (self.df["age"] / 365).round().astype(int)
        
        return self.df.copy()

    def feature_engineering_bmi(self):
        """Skapar ny akolumner som räknar ut BMI, rensar bort extremvärden och kategoriserar BMI-värden efter generella riktlinjer"""
        
        # Beräkna BMI
        self.df["bmi"] = round(self.df["weight"] / ((self.df["height"] / 100) ** 2), 2) 
        # Ta bort orimliga värden
        self.df = self.df.drop(self.df[(self.df["bmi"] > 80) | (self.df["bmi"] < 15)].index, axis=0)
        # Definiera vilkor och kategorier för BMI värden. Kodhjälp: https://medium.com/@heyamit10/implementing-pandas-np-select-f22ddd1706f6
        conditions = [
            (self.df["bmi"] >= 18.5) & (self.df["bmi"] < 25),
            (self.df["bmi"] >= 25) & (self.df["bmi"] < 30),
            (self.df["bmi"] >= 30) & (self.df["bmi"] < 35),
            (self.df["bmi"] >= 35) & (self.df["bmi"] < 40),
            (self.df["bmi"] >= 40)
        ]
        categories = [
            "Normal Range", 
            "Overweight", 
            "Obese (Class I)", 
            "Obese (Class II)", 
            "Obese (Class III)"
        ]
        self.df["bmi_cat"] = np.select(conditions, categories, default="Out of Range")
        return self.df.copy()

    def feature_engineering_bp(self):
        """Rensar blodtrycksdata och skapar en ny kolumn med blodtryckskategorier."""
        
        # Ta bort rader där diastoliskt tryck är större än systoliskt tryck
        self.df = self.df[self.df['ap_lo'] <= self.df['ap_hi']]
        # Ta bort orimliga värden för blodtryck 
        self.df = self.df[(self.df["ap_hi"] >= 90) & (self.df["ap_hi"] <= 200)]
        self.df = self.df[(self.df["ap_lo"] >= 60) & (self.df["ap_lo"] <= 110)]
        # Definiera villkor och kategorier för blodtryck
        conditions = [
            (self.df["ap_hi"] > 180) | (self.df["ap_lo"] > 120),
            (self.df["ap_hi"] >= 140) | (self.df["ap_lo"] >= 90),
            ((self.df["ap_hi"] >= 130) & (self.df["ap_hi"] <= 139)) | ((self.df["ap_lo"] >= 80) & (self.df["ap_lo"] <= 89)),
            ((self.df["ap_hi"] >= 120) & (self.df["ap_hi"] <= 129)) & (self.df["ap_lo"] < 80),
            (self.df["ap_hi"] < 120) & (self.df["ap_lo"] < 80)
        ]
        bp_categories = [
            "Hypertensive Crisis",
            "Hypertension Stage 2",
            "Hypertension Stage 1",
            "Elevated",
            "Normal"
        ]
        self.df["bp_cat"] = np.select(conditions, bp_categories, default="Out of Range")
        return self.df.copy()
    
    def get_clean_data(self):
        """Kör alla steg för databehandling i följd."""
        self.initial_data_proces()
        self.feature_engineering_bmi()
        self.feature_engineering_bp()
        return self.df.copy()


# -------------------------------
# Data Visualization
# -------------------------------
class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def eda_overview(self):
        """Skapar subplots med stapel & tårtdiagram samthistogram för att visualisera data."""
        
        # Skapa en figur med subplots
        fig, ax = plt.subplots(4, 2, figsize=(20, 15))
        
        # Countplot: CVD-distribution
        sns.countplot(x="CVD", data=self.df, palette="Set2", ax=ax[0, 0])
        ax[0, 0].set_title("Cardiovascular disease (CVD) distribution")
        ax[0, 0].set_xlabel("CVD")
        ax[0, 0].set_ylabel("Participants")
        # Countplot: CVD vs Gender
        sns.countplot(x="gender_label", data=self.df, hue="CVD", palette="Set2", ax=ax[0, 1])
        ax[0, 1].set_title("Cardiovascular disease (CVD) vs Gender")
        ax[0, 1].set_xlabel("Gender")
        ax[0, 1].set_ylabel("Participants")
        # Pie-chart: Kolesterolnivåer:
        chol_counts = self.df["cholesterol_label"].value_counts().sort_index()
        chol_labels = ["Normal", "Above normal", "Way above normal"]
        ax[1, 0].pie(chol_counts, labels=chol_labels, autopct="%1.1f%%", startangle=40, colors=["#66c2a5", "#8da0cb", "#fc8d62"])
        ax[1, 0].axis("equal")
        ax[1, 0].set_title("Cholesterol level distribution")
        # Pie-chart: Rökvanor
        smoke_counts = self.df["smoke_label"].value_counts()
        smoke_labels = smoke_counts.index.map({0: "No", 1: "Yes"})
        ax[1, 1].pie(smoke_counts, labels=smoke_labels, autopct="%1.1f%%", colors=["#66c2a5", "#8da0cb", "#fc8d62"])
        ax[1, 1].axis("equal")
        ax[1, 1].set_title("Smoking distribution of participants")
        # Histogram: Ålder (omvandlat till år)
        sns.histplot(self.df["age"], bins=50, kde=True, color="#66c2a5", ax=ax[2, 0])
        ax[2, 0].set_title("Age distribution for participants")
        ax[2, 0].set_xlabel("Age in years")
        # Histogram: Vikt
        sns.histplot(self.df["weight"], bins=50, kde=True, color="#66c2a5", ax=ax[2, 1])
        ax[2, 1].set_title("Weight distribution for participants")
        ax[2, 1].set_xlabel("Weight in kg")
        # Histogram: Längd
        sns.histplot(self.df["height"], bins=50, kde=True, color="#66c2a5", ax=ax[3, 0])
        ax[3, 0].set_title("Height distribution for participants")
        ax[3, 0].set_xlabel("Height in cm")
        # Countplot: CVD per ålder
        sns.countplot(x="age", hue="CVD", data=self.df, palette="Set2", ax=ax[3, 1])
        ax[3, 1].set_title("Cardiovascular Disease VS Age")
        ax[3, 1].set_xlabel("Age in years")
        
        # plotta alla subplots
        plt.tight_layout()
        plt.show()

    def disease_overwiev(self, figsize=(14, 24)):
        """Skapar en visuell översikt av samband mellan CVD och andra sjukdomar"""
        # Skapa en ny figur med subplots
        fig, ax = plt.subplots(4, 1, figsize=figsize)

        # Subplot 1: Rökna ut andel sjuka per blodtryck
        grouped = self.df.groupby(["bp_cat", "CVD"]).size().reset_index(name="count")
        totals = grouped.groupby("bp_cat")["count"].transform("sum") # Kodhjälp: https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby
        grouped["proportion"] = grouped["count"] / totals
        sns.barplot(
            x="bp_cat", 
            y="proportion", 
            hue="CVD", 
            data=grouped, 
            palette="Set2", # Färg-palett: https://seaborn.pydata.org/tutorial/color_palettes.html
            ax=ax[0],
            order=["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"]
        )
        ax[0].set_title("CVD per Blood Pressure Category")
        ax[0].set_xlabel("Blood pressure Category")
        ax[0].set_ylabel("Participants (proportion)")

        # Subplot 2: Räkna andel sjuka per BMI
        grouped_bmi = self.df.groupby(["bmi_cat", "CVD"]).size().reset_index(name="count")
        totals_bmi = grouped_bmi.groupby("bmi_cat")["count"].transform("sum")
        grouped_bmi["proportion"] = grouped_bmi["count"] / totals_bmi
        sns.barplot(
            x="bmi_cat", 
            y="proportion", 
            hue="CVD", 
            data=grouped_bmi, 
            palette="Set2", 
            ax=ax[1],
            order=["Normal Range", "Overweight", "Obese (Class I)", "Obese (Class II)", "Obese (Class III)"]
        )
        ax[1].set_title("CVD per BMI Category")
        ax[1].set_xlabel("BMI Category")
        ax[1].set_ylabel("Participants (proportion)")

        # Subplot 3: Räkna ut andel sjuka per kolesterolnivå
        grouped_chol = self.df.groupby(["cholesterol_label", "CVD"]).size().reset_index(name="count")
        totals_chol = grouped_chol.groupby("cholesterol_label")["count"].transform("sum")
        grouped_chol["proportion"] = grouped_chol["count"] / totals_chol
        sns.barplot(
            x="cholesterol_label", 
            y="proportion", 
            hue="CVD", 
            data=grouped_chol, 
            palette="Set2", 
            ax=ax[2],
            order=["Normal", "Above Normal", "Way Above Normal"]
        )
        ax[2].set_title("CVD per Cholesterol Category")
        ax[2].set_xlabel("Cholesterol Category")
        ax[2].set_ylabel("Participants (proportion)")

        # Subplot 4: Räkna ut andel sjuka per gluykoshalt
        grouped_gluc = self.df.groupby(["gluc_label", "CVD"]).size().reset_index(name="count")
        totals_gluc = grouped_gluc.groupby("gluc_label")["count"].transform("sum")
        grouped_gluc["proportion"] = grouped_gluc["count"] / totals_gluc
        sns.barplot(
            x="gluc_label", 
            y="proportion", 
            hue="CVD", 
            data=grouped_gluc, 
            palette="Set2", 
            ax=ax[3],
            order=["Normal", "Above Normal", "Way Above Normal"]
        )
        ax[3].set_title("CVD per Glucose Category")
        ax[3].set_xlabel("Glucose Category")
        ax[3].set_ylabel("Participants (proportion)")    

        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        """Visar en korrelationsmatris med en heatmap."""
        # Skapa en korrelationsmatris för att visa sambandet mellan fetaures 
        corr = self.df.select_dtypes(include=[np.number]).corr() # Kodhjälp: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.title("Correlation Matrix")
        plt.show()


# -------------------------------
# Data Preparer for Machine Learning Models
# -------------------------------
class DataPreparer:
    """
    Förbereder data för maskininlärning genom att skapa 2 olika dataset och dela upp datan i tränings-, validerings- och testset: 
    - Dataset 1 och 2 skapas genom att ta bort olika uppsättningar av features och skapa dummy-variabler.
    - 70 / 30 för träning och testning och sedan testdatan i 70 / 30 för validering och test. 
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def create_datasets(self):
        df_copy = self.df.copy()
        # Dataset 1: 
        df1 = df_copy.drop(columns=["ap_hi", "ap_lo", "height", "weight", "bmi", "cholesterol_label", "gluc_label", "smoke_label", "alco_label", "gender_label", "CVD"])
        df1 = pd.get_dummies(df1, columns=["bmi_cat", "bp_cat", "gender"])
        # Dataset 2: 
        df2 = df_copy.drop(columns=["bmi_cat", "bp_cat", "height", "weight" , "cholesterol_label", "gluc_label", "smoke_label", "alco_label", "gender_label", "CVD"])
        df2 = pd.get_dummies(df2, columns=["gender"])
        return df1, df2
        
    def train_val_test_split(self, df, target_col="cardio", test_size=0.3, val_size=0.3, random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state) # Kodhjälp: ChatGPT, prompt: "hur delar jag upp tränings, valideringsdata och testdata i en funktion?"
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------
# Model Training on Machine Learning Models and Evaluation 
# -------------------------------
class ModelTrainer:
    """Klass för att träna, utvärdera och spara olika maskininlärningsmodeller: Logistic Regression, Random Forest, Decision Tree och SVM."""
    
    def __init__(self):
        self.models = {}

        # Skapa pipeline för skalering, normailsering och modellering
    def train_logistic_regression(self, X_train, y_train, param_grid, cv=5): 
        pipe = Pipeline([ 
            ("standard", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("lr", LogisticRegression(max_iter=2500, random_state=42))
        ])
        # Korsvalidering och hyperparameter-tuning 
        grid = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Logistic Regression - Best Params:", grid.best_params_)
        print("Logistic Regression - CV Score:", grid.best_score_)
        self.models["log_reg"] = best_model
        return best_model
    
        # Träna Random Forest-modellen
    def train_random_forest(self, X_train, y_train, param_grid, cv=5):
        pipe = Pipeline([
            ("standard", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("rf", RandomForestClassifier(random_state=42))
        ])
        grid = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Random Forest - Best Params:", grid.best_params_)
        print("Random Forest - CV Score:", grid.best_score_)
        self.models["random_forest"] = best_model
        return best_model
    
        # Träna Decision Tree-modellen
    def train_decision_tree(self, X_train, y_train, param_grid, cv=5):
        pipe = Pipeline([
            ("standard", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("dt", DecisionTreeClassifier(random_state=42))
        ])
        grid = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Decision Tree - Best Params:", grid.best_params_)
        print("Decision Tree - CV Score:", grid.best_score_)
        self.models["decision_tree"] = best_model
        return best_model

        # Träna SVM-modellen 
    def train_svm(self, X_train, y_train, param_grid, cv=5, use_probabilities=False, sample_size=None): # Kodhjälp: ChatGPT, prompt: "Hur kan jag träna en SVM-modell med hyperparameter-tuning och korsvalidering utan att det tar över 8 h?"
        if sample_size is not None:
            X_train = X_train.sample(n=sample_size, random_state=42)
            y_train = y_train.loc[X_train.index]
        
        svm = SVC(probability=use_probabilities, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm)
        ])
        grid = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("SVM - Best Params:", grid.best_params_)
        print("SVM - CV Score:", grid.best_score_)
        self.models["svm"] = best_model
        return best_model
    
        # Spara ML-modellen till en fil 
    @staticmethod                                   
    def save_model(model, filename): # Kodhjälp: ChatGPT, prompt: "Hur sparar jag resultatet från en modell jag kört, så jag slipper köra om den längre fram, tex min SVM?"
        """Sparar en tränad modell till en fil."""
        joblib.dump(model, filename)
        print(f"Modellen sparad som: {filename}")
    
    @staticmethod  
    def load_model(filename):
        """Laddar en modell från en fil."""
        model = joblib.load(filename)
        print(f"Modellen laddad från: {filename}")
        return model

        # Utvärdera modellen och skriv ut en klassifikatonsrapport
    def evaluate_model(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        print("Classification Report:\n", classification_report(y_val, y_pred))
        return y_pred

    def final_evaluation_on_test(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Tränar vald ML-modell på hela träningsdatan (train + val) och utvärderar på testdatan.
        Skriver ut klassifikations rapport samt förvirrings matris.
        """
        # Kombinera train + val
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # Träna på sammanslagen data
        model.fit(X_train_val, y_train_val)

        # Gör prediktioner
        y_pred = model.predict(X_test)
        
        # Klassifikationsrapport
        print("Final Evaluation on Test Data")
        print(classification_report(y_test, y_pred))

        # Skapa och visa förvirringsmatris
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        cm_display.ax_.set_title("Confusion Matrix")
        plt.show()
        
    def show_evaluation_table(self, models: dict, X_val_list, y_val_list, dataset_names, sort_by="Recall", highlight=False): # Kodhjälp: ChatGPT, prompt: "Hur kan jag skapa en tabell för precision, recall och f1-score för varje modell för båda datasetten?"
        """
        Visar en tabell med precision, recall och F1-score för varje modell i varje dataset.

        :param models: Dict med modellnamn som nycklar och modeller som värden.
        :param X_val_list: Lista med X_val för respektive dataset.
        :param y_val_list: Lista med y_val för respektive dataset.
        :param dataset_names: Lista med namn för dataset (ex: ['Dataset 1', 'Dataset 2'])
        :param sort_by: Vilket mått som ska sorteras på ("Recall", "Precision", "F1 Score").
        :param highlight: Om True används Pandas Styler för att highlighta bästa värden.
        """
        results = []
        for ds_index, (X_val, y_val) in enumerate(zip(X_val_list, y_val_list)):
            for name, model in models.items():
                y_pred = model.predict(X_val)
                precision = precision_score(y_val, y_pred, pos_label=1)
                recall = recall_score(y_val, y_pred, pos_label=1)
                f1 = f1_score(y_val, y_pred, pos_label=1)
                results.append({
                    "Dataset": dataset_names[ds_index],
                    "Model": name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                })

        df = pd.DataFrame(results)
        df[["Precision", "Recall", "F1 Score"]] = df[["Precision", "Recall", "F1 Score"]].round(2)
        df_sorted = df.sort_values(by=sort_by, ascending=False)

        if highlight:
            styled = df_sorted.style.highlight_max(color="lightgreen", axis=0, subset=["Precision", "Recall", "F1 Score"])
            display(styled)
        else:
            display(df_sorted)


# -------------------------------
# Ensemble Modellering (Voting Classifier)
# -------------------------------
class EnsembleModeler:
    def __init__(self):
        self.voting_model = None

    @staticmethod
    def get_best_models(dataset, use_probabilities=False):
        """Returnerar de bästa modellerna med rätt SVM-inställning beroende på voting-typ."""
        if dataset == 1:
            log = LogisticRegression(C=10, penalty="l1", solver="liblinear", max_iter=2500)
            tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=2)
            rnd = RandomForestClassifier(max_depth=10, min_samples_leaf=4,
                                         min_samples_split=10, n_estimators=300)
            svm = SVC(C=1, gamma="scale", kernel="rbf", probability=use_probabilities)
        elif dataset == 2:
            log = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", max_iter=2500)
            tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, min_samples_split=2)
            rnd = RandomForestClassifier(max_depth=10, min_samples_leaf=2,
                                         min_samples_split=10, n_estimators=200)
            svm = SVC(C=1, gamma="scale", kernel="rbf", probability=use_probabilities)
        else:
            raise ValueError("Dataset must be 1 or 2")
        return log, tree, rnd, svm

    def create_voting_classifier(self, dataset, vote_type, X_train, y_train):
        """
        Skapar och tränar en VotingClassifier (hard/soft) beroende på användning.
        """
        use_proba = (vote_type == "soft")
        log, tree, rnd, svm = self.get_best_models(dataset, use_probabilities=use_proba)
        
        self.voting_model = VotingClassifier(estimators=[
            ('log', log),
            ('tree', tree),
            ('rnd', rnd),
            ('svm', svm)
        ], voting=vote_type)

        self.voting_model.fit(X_train, y_train)
        return self.voting_model

    def evaluate_voting_classifier(self, X_val, y_val):
        """
        Utvärderar VotingClassifier och skriver ut classification report.
        """
        if self.voting_model is None:
            raise ValueError("The voting classifier has not been created and trained yet.")
        y_pred = self.voting_model.predict(X_val)
        print("Voting Classifier Evaluation:\n", classification_report(y_val, y_pred))
        return y_pred
    
    @staticmethod
    def save_model(model, filename):
        """Sparar tränad modell till fil."""
        joblib.dump(model, filename)
        print(f"Modellen sparad som: {filename}")

    def load_model(filename):
        """Laddar modell från fil."""
        model = joblib.load(filename)
        print(f"Modellen laddad från: {filename}")
        return model
