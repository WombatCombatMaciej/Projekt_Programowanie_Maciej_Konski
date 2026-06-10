from matplotlib.pylab import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sympy import re
import category_encoders as ce
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import chi2_contingency,loguniform, randint

baza = pd.read_csv("AgeDataset-V1-Part1.csv")

#Preprocessing danycb
#Tylko ludzie urodzeni po 1900 roku
baza = baza[baza['Birth year']>=1900]
baza = baza[baza["Age of death"].between(15, 120)]
#Przefiltrowanie innych płci ze wzgłedu na ich małą ilość w zbiorze danych
baza = baza[baza["Gender"].isin(["Male", "Female"])]
#Zamiana płci na wartości liczbowe
mapping = {'Male': 0, 'Female': 1}
baza["IsFemale"] = baza["Gender"].map(mapping)
#Usunięcie pustych wartości
baza = baza.dropna(subset=["Age of death", "Birth year", "Gender"])
#Kodowanie zawodu
baza["Occupation_code"] = baza["Occupation"].str.strip().str.lower()

encoder = ce.BinaryEncoder(cols=['Occupation_code'])
baza = encoder.fit_transform(baza)

kolumny_zawodow = [col for col in baza.columns if col.startswith('Occupation_code_')]
#print("Wygenerowane kolumny dla zawodów:", kolumny_zawodow)

#print(f"Liczba duplikatów: {baza.duplicated().sum()}")
#print(baza.isnull().sum())
#Standardyzacja nazw państw i usunięcie nieprawidłowych wartości wieku
baza["Country"] = baza["Country"].str.strip().str.title()
baza = baza[(baza["Age of death"] >= 0) & (baza["Age of death"] <= 120)]
baza = baza[baza["Death year"] >= baza["Birth year"]]
baza = baza.dropna().reset_index(drop=True)
print(baza.shape[0])

mmsc = MinMaxScaler()
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
    df = df[['Birth year', 'Death year', 'Age of death', 'job_safety_level',"IsFemale", "Occupation_death_mean_age"]]
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


def DobieranieParametrow(model,y:pd.Series,X:pd.Series):

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

  
        param_dist = {
            "C": loguniform(1e-3, 1e2),
            "solver": ["lbfgs", "saga"],
        }
    elif(model == "DecisionTree"):
        model_lr = DecisionTreeClassifier(random_state=42)

        param_dist = {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1,5),
        }
    elif(model =="LinearRegression"):
        model_lr = LinearRegression(
            
        )


    random_search = RandomizedSearchCV(
        estimator=model_lr, 
        param_distributions=param_dist,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    random_search.fit(X_train_std, y_train)

    print(f"Najlepsze parametry: {random_search.best_params_}")
    print(f"Najlepszy wynik na walidacji: {random_search.best_score_:.2f}")


def RegresjaLogistyczna(y:pd.Series, X:pd.Series,c=0.01,solver="saga",) -> pd.Series:

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

def Drzewo(df) -> pd.Series:
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

def RegresjaLiniowa(y:pd.Series, X:pd.Series) -> pd.Series:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sc = MinMaxScaler()

    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    lr = LinearRegression()

    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    return (y_test, y_pred)

def Przedziały(kat1:pd.Series, kat2:pd.Series, Max_depth=5,Min_samples_leaf=2,Min_samples_split=5) -> list:
    X_tree = kat1
    y_tree = kat2
    drzewo = DecisionTreeClassifier(max_depth=Max_depth,min_samples_leaf=Min_samples_leaf,min_samples_split=Min_samples_split, random_state=42, class_weight='balanced')
    drzewo.fit(X_tree, y_tree)
    wewnetrzne_progi = drzewo.tree_.threshold
    progi = sorted(list(set([round(float(p), 2) for p in wewnetrzne_progi if p != -2])))
    print("Wyciągnięte progi:", progi)
    #new_column = pd.cut(df[kat1], bins=[-1] + progi + [float('inf')], labels=False)
    #print(df[["Age of death", "AgeGroup"]].head(10))
    return progi
    #print(export_text(drzewo, feature_names=["Age of death"]))

    #return (progi)

def Tree_Pred(y:pd.Series, X:pd.Series) -> pd.Series:

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




def KorelacjaCramera(df, kolumna1="Occupation", kolumna2="Manner of death"):
    # 1. Tworzymy tabelę krzyżową (częstości występowania)
    tabela_krzyzowa = pd.crosstab(df[kolumna1], df[kolumna2])

    # 2. Odpalamy test Chi-kwadrat
    chi2, p_val, dof, expected = chi2_contingency(tabela_krzyzowa)

    # 3. Obliczamy współczynnik V-Cramera
    n = tabela_krzyzowa.sum().sum()
    r, c = tabela_krzyzowa.shape
    v_cramera = np.sqrt(chi2 / (n * min(r - 1, c - 1)))

    print(f"--- Wyniki analizy dla {kolumna1} vs {kolumna2} ---")
    print(f"p-value: {p_val:.5f}")
    if p_val < 0.05:
        print("-> Istnieje statystycznie istotna zależność między zawodem a powodem śmierci!")
    else:
        print("-> Brak istotnych dowodów na zależność.")
    print(f"Współczynnik V-Cramera (siła związku): {v_cramera:.3f}")




baza["Occupation_death_mean_age"] = baza.groupby("Occupation")["Age of death"].transform("mean").round(1)
baza["Occupation_death_mean_age_scaled"] = mmsc.fit_transform(baza[["Occupation_death_mean_age"]])
baza = baza.dropna(subset=["Occupation_death_mean_age"])
#print(baza["Occupation_death_mean_age_scaled"].head(5).round(2))

koszyki = [-float('inf'), 40, 66, float('inf')]
etykiety = [0, 1, 2]


baza["Death Age Group"]= pd.cut(baza["Age of death"], bins=koszyki, labels=etykiety)
baza["job_safety_level"] = pd.cut(baza["Occupation_death_mean_age"], bins=koszyki, labels=etykiety)


Walidacja(RegresjaLiniowa(y=baza["Age of death"], X=baza[["Occupation_death_mean_age_scaled", "Birth year"]]), test_type="regression")
#DobieranieParametrow(model="LogisticRegression", y=baza["Age of death"], X=baza[["Occupation_death_mean_age_scaled"]])
#Macierz(baza)
#DobieranieParametrow(model="DecisionTree", y=baza[["Age of death"]], X=baza[["Birth year"]])
Walidacja(Tree_Pred(baza[["Death Age Group"]], baza[["Birth year"]]), test_type="classification")
#baza.to_csv("Preprocessed_AgeDataset.csv", index=False)
#print(baza.columns)
#baza.head
#GestosPopulacji(baza)