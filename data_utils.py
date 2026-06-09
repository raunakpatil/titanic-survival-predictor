import pandas as pd
import numpy as np


def load_data():
    """Load Titanic data, auto-downloading from public sources if not present."""
    import os
    import urllib.request

    os.makedirs("data", exist_ok=True)

    _TRAIN_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    _TEST_URL  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/titanic.csv"

    # ── train.csv ──────────────────────────────────────────────────────────────
    if not os.path.exists("data/train.csv"):
        # Primary source: datasciencedojo — full 891-row labelled dataset
        urllib.request.urlretrieve(_TRAIN_URL, "data/train.csv")

    # ── test.csv ───────────────────────────────────────────────────────────────
    # We don't need ground-truth labels for test; generate a small synthetic
    # stand-in from the training data so engineer_features() always gets two dfs.
    if not os.path.exists("data/test.csv"):
        _tmp = pd.read_csv("data/train.csv")
        # Use the last 20% as a pseudo-test set (no Survived column)
        _test = _tmp.tail(int(len(_tmp) * 0.2)).drop(columns=["Survived"], errors="ignore").copy()
        _test.to_csv("data/test.csv", index=False)

    train_df = pd.read_csv("data/train.csv")
    test_df  = pd.read_csv("data/test.csv")
    return train_df, test_df


def _extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    rare = {"Lady", "Countess", "Capt", "Col", "Don", "Rev",
            "Dr", "Major", "Sir", "Jonkheer", "Dona"}
    if title in rare:
        return "Rare"
    elif title in ("Mlle", "Ms"):
        return "Miss"
    elif title == "Mme":
        return "Mrs"
    return title


def _title_to_int(title):
    mapping = {"Mr": 1, "Miss": 2, "Mrs": 2, "Master": 4, "Rare": 3}
    return mapping.get(title, 0)


def engineer_features(train_df, test_df):
    combined = pd.concat([train_df.drop("Survived", axis=1), test_df], axis=0).reset_index(drop=True)

    combined["Age"] = combined.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["Embarked"] = combined["Embarked"].fillna("S")

    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)
    combined["Title"] = combined["Name"].apply(_extract_title).apply(_title_to_int)
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]

    combined["Sex"] = (combined["Sex"] == "female").astype(int)
    embarked_dummies = pd.get_dummies(combined["Embarked"], prefix="Embarked")
    combined = pd.concat([combined, embarked_dummies], axis=1)

    feature_names = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked_C", "Embarked_Q", "Embarked_S",
        "FamilySize", "IsAlone", "Title", "FarePerPerson"
    ]

    for col in feature_names:
        if col not in combined.columns:
            combined[col] = 0

    n_train = len(train_df)
    X_train = combined.iloc[:n_train][feature_names].values
    y_train = train_df["Survived"].values
    X_test = combined.iloc[n_train:][feature_names].values

    return X_train, y_train, X_test, feature_names


def get_historical_twin(train_df, sex, pclass, age):
    subset = train_df[(train_df["Sex"] == sex) & (train_df["Pclass"] == pclass)].copy()
    if subset.empty:
        return None
    subset = subset.dropna(subset=["Age"])
    if subset.empty:
        return None
    subset["age_diff"] = (subset["Age"] - age).abs()
    twin = subset.sort_values("age_diff").iloc[0]
    return twin


def search_passengers(train_df, query):
    """Search passengers by name (case-insensitive substring match)."""
    if not query or len(query.strip()) < 2:
        return pd.DataFrame()
    mask = train_df["Name"].str.contains(query.strip(), case=False, na=False)
    return train_df[mask].copy()
