from matplotlib.pylab import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sympy import re
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text

baza = pd.read_csv("AgeDataset-V1-Part1.csv")

#Preprocessing danycb
#Tylko ludzie urodzeni po 1900 roku
baza = baza[baza['Birth year']>=1900]
#Przefiltrowanie innych płci ze wzgłedu na ich małą ilość w zbiorze danych
baza = baza[baza["Gender"].isin(["Male", "Female"])]
#Zamiana płci na wartości liczbowe
mapping = {'Male': 0, 'Female': 1}
baza["IsFemale"] = baza["Gender"].map(mapping)
#Usunięcie pustych wartości
baza = baza.dropna(subset=["Age of death", "Birth year", "Gender"])
#Kodowanie zawodu
baza["Occupation"] = baza["Occupation"].str.strip().str.lower()
Zawody = ["Occupation_0",
            "Occupation_1",
            "Occupation_2",
            "Occupation_3",
            "Occupation_4",
            "Occupation_5",
            "Occupation_6",
            "Occupation_7",
            "Occupation_8",
            "Occupation_9",
            "Occupation_10"]

encoder = ce.BinaryEncoder(cols=['Occupation'])
baza = encoder.fit_transform(baza)

#print(f"Liczba duplikatów: {baza.duplicated().sum()}")
#print(baza.isnull().sum())
#Standardyzacja nazw państw i usunięcie nieprawidłowych wartości wieku
baza["Country"] = baza["Country"].str.strip().str.title()
baza = baza[(baza["Age of death"] >= 0) & (baza["Age of death"] <= 120)]
baza = baza[baza["Death year"] >= baza["Birth year"]]

#Histogram danych
def Histogram(df):
    
    df = df[['Gender', 'Country', 'Occupation',
       'Birth year', 'Death year', 'Manner of death', 'Age of death']]
    df.hist(bins=50, figsize=(20,15))
    plt.show()
#Wykres Słupkowy pokazujący wiek zgonu w zależności od płci
def WykresSlupkowyPlec(df):
    male_age = df[df['Gender'] == 'Male']['Age of death'].dropna()
    female_age = df[df['Gender'] == 'Female']['Age of death'].dropna()
    plt.violinplot([male_age, female_age], showmeans=True)
    plt.xticks([1,2], ['Male', 'Female'])
    plt.ylabel('Age of Death')
    plt.title('Distribution of Age of Death')
    plt.show()

def GestosPopulacji(df):
    counts = df.groupby('Country').size().reset_index(name="Ilosc")
    print(counts.head(10))
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world_map = world.merge(counts, left_on="NAME", right_on="Country", how="left")
    world_map["Ilosc"] = world_map["Ilosc"].fillna(0)
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    world.plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.5)
    world_map.plot(
        column="Ilosc",
        ax=ax,
        legend=True,
        cmap="OrRd",
        legend_kwds={
            "label": "Liczba osób w zbiorze danych",
            "orientation": "horizontal",
        },
        edgecolor="black",
        linewidth=0.2,
    )
    plt.title("Gęstość występowania państw w zbiorze danych", fontsize=16)
    ax.set_axis_off()
    plt.show()

#Tu do poprawki bo ta korelacja jest bez sensu na razie
def Macierz(df):
    #Tu do poprawki bo ta korelacja jest bez sensu na razie
    df = df[['Birth year', 'Death year', 'Age of death', 'Occupation Code']]
    macierz_korelacji = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        macierz_korelacji,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt = ".2f",
        linewidths=0.5,
    )
    plt.title("Macierz korelacji")
    plt.show()


def DobieranieParametrow(df,model,x_data,y_data):
    df = df[df["Age of death"].between(15, 120)]
    X = df[
        x_data
    ]

    y = pd.cut(
        df[y_data], bins=[0, 9.5, 41.5, 120], labels=[0, 1, 2], include_lowest=True
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    sc = MinMaxScaler()

    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    if(model == "LogisticRegression"):
        model_lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )

  
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga"],
        }
    elif(model == "DDecisionTree"):
        model_lr = DecisionTreeClassifier(random_state=42)

        param_grid = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }


    grid_search = GridSearchCV(
        estimator=model_lr, 
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train_std, y_train)

    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepszy wynik na walidacji: {grid_search.best_score_:.2f}")


def Regresja(df):
    df = df[df["Age of death"].between(15, 120)]
    X = df[
        [
            "Occupation_0",
            "Occupation_1",
            "Occupation_2",
            "Occupation_3",
            "Occupation_4",
            "Occupation_5",
            "Occupation_6",
            "Occupation_7",
            "Occupation_8",
            "Occupation_9",
            "Occupation_10",
            'Birth year',
            "IsFemale"
        ]
    ]

    y = pd.qcut(
        df["Age of death"], q=3, labels=[0, 1, 2]
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    sc = MinMaxScaler()

    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)


    re = LogisticRegression(
        C=0.01,
        solver="saga",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        
    )


    re.fit(X_train_std, y_train)
    y_pred = re.predict(X_test_std)
    return (y_test, y_pred)

def Drzewo(df):
    df = df[df["Age of death"].between(15, 120)]
    X = df[
        [
            "Occupation_0",
            "Occupation_1",
            "Occupation_2",
            "Occupation_3",
            "Occupation_4",
            "Occupation_5",
            "Occupation_6",
            "Occupation_7",
            "Occupation_8",
            "Occupation_9",
            "Occupation_10",
            'Birth year',
            "IsFemale"
        ]
    ]
    y = pd.qcut(
        df["Age of death"], q=3, labels=[0, 1, 2]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=5, min_samples_leaf=2, class_weight='balanced')
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    return (y_test, y_pred)

def PrzedziałyWiekowe(df,kat1, kat2,tree_depth=5):
    X_tree = df[[kat1]]
    y_tree = df[kat2]
    drzewo = DecisionTreeClassifier(max_depth=tree_depth, random_state=42, class_weight='balanced')
    drzewo.fit(X_tree, y_tree)
    wewnetrzne_progi = drzewo.tree_.threshold
    progi_wieku = sorted(list(set([round(float(p), 2) for p in wewnetrzne_progi if p != -2])))
    print("Wyciągnięte progi:", progi_wieku)
    df["AgeGroup"] = pd.cut(df[kat1], bins=[-1] + progi_wieku + [float('inf')], labels=False)
    #print(df[["Age of death", "AgeGroup"]].head(10))

    #print(export_text(drzewo, feature_names=["Age of death"]))

    return (progi_wieku)

def DeathPred(df,progi_wieku,kat1="Age of death"):
    df["AgeGroup"] = pd.cut(df[kat1], bins=[-1] + progi_wieku + [float('inf')], labels=False)
    print(df[["Age of death", "AgeGroup"]].head(10))
    X = df[
        [
            "Occupation_0",
            "Occupation_1",
            "Occupation_2",
            "Occupation_3",
            "Occupation_4",
            "Occupation_5",
            "Occupation_6",
            "Occupation_7",
            "Occupation_8",
            "Occupation_9",
            "Occupation_10",
            'Birth year',
            "IsFemale"
        ]
    ]
    y = df["AgeGroup"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=5, min_samples_leaf=2, class_weight='balanced')
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    return (y_test, y_pred)

def Walidacja(y_test_pred, test_type="classification"):
    y_test = y_test_pred[0]
    y_pred = y_test_pred[1]
    if test_type == "classification":
        clsrp = classification_report(y_test, y_pred)
        print(clsrp)
    elif test_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")


Walidacja(Drzewo(baza),test_type="regression")