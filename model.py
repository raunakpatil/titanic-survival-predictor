import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    # Cross-validated accuracy (5-fold) - more honest than train accuracy
    cv_scores = cross_val_score(
        XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        X_train, y_train, cv=5, scoring="accuracy"
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return model, explainer, cv_mean, cv_std


def predict_survival(model, explainer, sex, pclass, age, fare, sibsp, parch, embarked, feature_names):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    sex_enc = 1 if sex == "female" else 0
    embarked_c = 1 if embarked == "C" else 0
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0

    if sex == "female":
        title = 2
    elif age < 16:
        title = 4
    else:
        title = 1

    fare_per_person = fare / max(family_size, 1)

    row = {
        "Pclass": pclass, "Sex": sex_enc, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Embarked_C": embarked_c, "Embarked_Q": embarked_q, "Embarked_S": embarked_s,
        "FamilySize": family_size, "IsAlone": is_alone,
        "Title": title, "FarePerPerson": fare_per_person,
    }

    input_df = pd.DataFrame([row])[feature_names]
    prob = model.predict_proba(input_df)[0][1]
    shap_vals_raw = explainer(input_df).values[0]

    readable = {
        "Pclass": "Ticket class", "Sex": "Gender", "Age": "Age",
        "SibSp": "Siblings/spouses", "Parch": "Parents/children", "Fare": "Fare paid",
        "Embarked_C": "Embarked Cherbourg", "Embarked_Q": "Embarked Queenstown",
        "Embarked_S": "Embarked Southampton", "FamilySize": "Family size",
        "IsAlone": "Travelling alone", "Title": "Title (Mr/Mrs/Master)", "FarePerPerson": "Fare per person",
    }

    shap_dict = {readable.get(f, f): round(float(v), 4)
                 for f, v in zip(feature_names, shap_vals_raw)}
    shap_dict = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6])

    return prob, shap_dict, input_df


def get_shap_values(explainer, X):
    return explainer(X)


def build_passenger_features(sex, pclass, age, fare, sibsp, parch, embarked, feature_names):
    """Build a feature row dict for a single passenger."""
    family_size = sibsp + parch + 1
    sex_enc = 1 if sex == "female" else 0
    title = 2 if sex == "female" else (4 if age < 16 else 1)
    row = {
        "Pclass": pclass, "Sex": sex_enc, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Embarked_C": 1 if embarked == "C" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
        "FamilySize": family_size, "IsAlone": 1 if family_size == 1 else 0,
        "Title": title, "FarePerPerson": fare / max(family_size, 1),
    }
    return pd.DataFrame([row])[feature_names]


def run_survival_simulator(model, train_df, sex, pclass, age, feature_names, n=100):
    """Sample similar passengers and return predicted survival probs."""
    subset = train_df[
        (train_df["Sex"] == sex) &
        (train_df["Pclass"] == pclass) &
        (train_df["Age"].between(max(1, age - 15), min(80, age + 15)))
    ].copy()

    if len(subset) < 5:
        subset = train_df[(train_df["Sex"] == sex) & (train_df["Pclass"] == pclass)].copy()

    if subset.empty:
        return None

    sample = subset.sample(n=min(n, len(subset)), replace=True, random_state=42)
    probs = []
    for _, r in sample.iterrows():
        age_v = r["Age"] if not pd.isna(r["Age"]) else age
        fare_v = r["Fare"] if not pd.isna(r["Fare"]) else 30
        emb = r.get("Embarked") or "S"
        fs = int(r.get("SibSp", 0) or 0) + int(r.get("Parch", 0) or 0) + 1
        sex_enc = 1 if sex == "female" else 0
        title = 2 if sex == "female" else (4 if age_v < 16 else 1)
        feat = {
            "Pclass": pclass, "Sex": sex_enc, "Age": age_v,
            "SibSp": int(r.get("SibSp", 0) or 0), "Parch": int(r.get("Parch", 0) or 0),
            "Fare": fare_v, "Embarked_C": 1 if emb == "C" else 0,
            "Embarked_Q": 1 if emb == "Q" else 0, "Embarked_S": 1 if emb == "S" else 0,
            "FamilySize": fs, "IsAlone": 1 if fs == 1 else 0,
            "Title": title, "FarePerPerson": fare_v / max(fs, 1),
        }
        row_df = pd.DataFrame([feat])[feature_names]
        p = model.predict_proba(row_df)[0][1]
        probs.append(p)
    return probs
