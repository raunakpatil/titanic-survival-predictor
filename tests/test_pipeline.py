import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils import (FEATURE_NAMES, build_features, clean_data, load_data,  # noqa: E402
                        passenger_frame, search_passengers)
from model import explain_passenger, make_explainer, run_survival_simulator, train_model  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    df = clean_data(load_data())
    X = build_features(df)
    y = df["Survived"].to_numpy()
    model, oof_probs, cv_mean, cv_std = train_model(X, y)
    return df, X, y, model, make_explainer(model, X), oof_probs, cv_mean


def test_features_complete(trained):
    _, X, *_ = trained
    assert list(X.columns) == FEATURE_NAMES
    assert not X.isna().any().any()


def test_inference_features_match_training():
    # An adult "Mr" with no name must get exactly the same features as in training.
    df = clean_data(load_data())
    row = df[df["Name"].str.contains(r"\bMr\.", regex=True) & (df["Age"] >= 16)].iloc[[0]]
    made_up = passenger_frame(*row[["Sex", "Pclass", "Age", "Fare", "SibSp", "Parch", "Embarked"]].iloc[0])
    np.testing.assert_array_equal(build_features(row).to_numpy(), build_features(made_up).to_numpy())


def test_cv_accuracy_reasonable(trained):
    *_, oof_probs, cv_mean = trained
    assert 0.78 < cv_mean < 0.90
    assert ((oof_probs >= 0) & (oof_probs <= 1)).all()


@pytest.mark.parametrize("profile", [
    ("female", 1, 30, 80, 1, 0, "C"),
    ("male", 3, 25, 8, 0, 0, "S"),
    ("male", 2, 6, 25, 1, 2, "Q"),
])
def test_shap_contributions_sum_to_probability(trained, profile):
    _, _, _, model, explainer, *_ = trained
    prob, base, contribs = explain_passenger(model, explainer, passenger_frame(*profile))
    assert 0 <= prob <= 1
    assert base + sum(contribs.values()) == pytest.approx(prob, abs=1e-4)


def test_simulator(trained):
    df, _, _, model, *_ = trained
    probs = run_survival_simulator(model, df, "female", 2, 30, n=100)
    assert 5 <= len(probs) <= 100
    assert ((probs >= 0) & (probs <= 1)).all()


def test_search_is_literal_and_case_insensitive():
    df = load_data()
    assert len(search_passengers(df, "astor")) > 0
    assert len(search_passengers(df, "(Madeleine")) == 1  # "(" is not treated as regex
    assert search_passengers(df, "x").empty  # too short


def test_sinking_timeline():
    import history
    assert history.TOTAL_SEATS == 1178 and len(history.LIFEBOATS) == 20
    assert history.clock(0) == "23:40" and history.clock(history.END) == "02:20"
    start, end = history.state_at(0), history.state_at(history.END)
    assert start["boats"] == 0 and end["boats"] == 20
    assert end["pitch"] > start["pitch"] and end["sink"] > start["sink"]


def test_ship_viewer_assets_present():
    root = Path(__file__).resolve().parents[1] / "ship_viewer"
    for name in ("index.html", "ship.js", "titanic.glb", "vendor/three.module.min.js", "vendor/GLTFLoader.js"):
        assert (root / name).exists(), name
