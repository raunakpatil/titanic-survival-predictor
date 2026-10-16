import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from data_utils import READABLE_FEATURES, build_features


def make_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


def train_model(X, y, n_splits=5):
    """Fit the final model and score it honestly with stratified k-fold CV.

    Returns the fitted model, out-of-fold survival probabilities (each passenger
    predicted by a model that never saw them) and per-fold accuracy mean/std.
    """
    oof_probs = np.zeros(len(y))
    fold_acc = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in folds.split(X, y):
        fold_model = make_model().fit(X.iloc[train_idx], y[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_acc.append(((oof_probs[val_idx] > 0.5) == y[val_idx]).mean())

    model = make_model().fit(X, y)
    return model, oof_probs, float(np.mean(fold_acc)), float(np.std(fold_acc))


def make_explainer(model, X, n_background=100):
    """SHAP explainer whose values are in probability space.

    The base value plus all feature contributions sums exactly to the predicted
    survival probability, so they can be drawn on a 0-100% axis.
    """
    background = shap.sample(X, n_background, random_state=42)
    return shap.TreeExplainer(
        model, background,
        model_output="probability", feature_perturbation="interventional",
    )


def explain_passenger(model, explainer, passenger_df, top_n=6):
    """Predict one passenger and break the prediction into SHAP contributions.

    Returns (probability, base_rate, contributions) where contributions holds
    the ``top_n`` largest features by readable name plus an "Other features"
    bucket, so base_rate + sum(contributions) == probability.
    """
    X = build_features(passenger_df)
    prob = float(model.predict_proba(X)[0, 1])
    exp = explainer(X)
    base = float(np.ravel(exp.base_values)[0])

    contribs = pd.Series(exp.values[0], index=X.columns).rename(READABLE_FEATURES)
    contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    top = contribs.iloc[:top_n].to_dict()
    rest = contribs.iloc[top_n:].sum()
    if abs(rest) > 1e-4:
        top["Other features"] = float(rest)
    return prob, base, top


def global_importance(explainer, X):
    """Mean |SHAP| per feature, in percentage points of survival probability."""
    values = explainer(X).values
    return pd.Series(np.abs(values).mean(axis=0), index=X.columns).rename(READABLE_FEATURES)


def run_survival_simulator(model, df, sex, pclass, age, n=100, age_window=15):
    """Predicted survival for up to ``n`` real passengers similar to the user."""
    subset = df[(df["Sex"] == sex) & (df["Pclass"] == pclass)]
    similar = subset[subset["Age"].between(age - age_window, age + age_window)]
    if len(similar) >= 5:
        subset = similar
    if subset.empty:
        return None
    sample = subset.sample(n=min(n, len(subset)), random_state=42)
    return model.predict_proba(build_features(sample))[:, 1]
