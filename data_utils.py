from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
_TRAIN_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

FEATURE_NAMES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked_C", "Embarked_Q", "Embarked_S",
    "FamilySize", "IsAlone", "Title", "FarePerPerson",
]

READABLE_FEATURES = {
    "Pclass": "Ticket class", "Sex": "Gender", "Age": "Age",
    "SibSp": "Siblings/spouses", "Parch": "Parents/children", "Fare": "Fare paid",
    "Embarked_C": "Embarked Cherbourg", "Embarked_Q": "Embarked Queenstown",
    "Embarked_S": "Embarked Southampton", "FamilySize": "Family size",
    "IsAlone": "Travelling alone", "Title": "Title (Mr/Mrs/Master)",
    "FarePerPerson": "Fare per person",
}

PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}


def load_data():
    """Load the labelled 891-passenger Titanic set, downloading it on first run."""
    if not TRAIN_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        try:
            urllib.request.urlretrieve(_TRAIN_URL, TRAIN_PATH)
        except OSError as exc:
            TRAIN_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download the Titanic dataset ({exc}). "
                f"Place Kaggle's train.csv at {TRAIN_PATH} and restart."
            ) from exc
    return pd.read_csv(TRAIN_PATH)


def clean_data(df):
    """Fill missing Age (median by class & sex), Fare and Embarked."""
    df = df.copy()
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


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


def build_features(df):
    """Turn passenger rows into the model's feature matrix.

    Used for both training and inference so the two can never drift apart.
    ``df`` needs Sex ("male"/"female"), Pclass, Age, Fare, SibSp, Parch and
    Embarked. Title comes from Name when present; for made-up passengers it is
    inferred from sex and age (Mrs/Miss, Master for boys under 16, else Mr).
    """
    out = pd.DataFrame(index=df.index)
    out["Pclass"] = df["Pclass"].astype(int)
    out["Sex"] = (df["Sex"] == "female").astype(int)
    out["Age"] = df["Age"].astype(float)
    out["SibSp"] = df["SibSp"].astype(int)
    out["Parch"] = df["Parch"].astype(int)
    out["Fare"] = df["Fare"].astype(float)
    for port in PORTS:
        out[f"Embarked_{port}"] = (df["Embarked"] == port).astype(int)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    if "Name" in df:
        out["Title"] = df["Name"].map(_extract_title).map(_title_to_int)
    else:
        out["Title"] = np.select([out["Sex"] == 1, out["Age"] < 16], [2, 4], default=1)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    return out[FEATURE_NAMES].astype(float)


def passenger_frame(sex, pclass, age, fare, sibsp, parch, embarked):
    """A single made-up passenger as a one-row frame ready for build_features."""
    return pd.DataFrame([{
        "Sex": sex, "Pclass": pclass, "Age": age, "Fare": fare,
        "SibSp": sibsp, "Parch": parch, "Embarked": embarked,
    }])


def get_historical_twin(df, sex, pclass, age):
    """Real passenger of the same sex and class whose (recorded) age is closest."""
    subset = df[(df["Sex"] == sex) & (df["Pclass"] == pclass)].dropna(subset=["Age"])
    if subset.empty:
        return None
    return subset.loc[(subset["Age"] - age).abs().idxmin()]


def search_passengers(df, query):
    """Search passengers by name (case-insensitive substring match)."""
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    mask = df["Name"].str.contains(query.strip(), case=False, na=False, regex=False)
    return df[mask]
