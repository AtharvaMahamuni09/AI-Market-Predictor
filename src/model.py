\
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, roc_auc_score
)
import joblib
import os

@dataclass
class TrainedModels:
    regressor: GradientBoostingRegressor
    classifier: LogisticRegression
    scaler: StandardScaler
    feature_cols: list

def time_split(X, y_reg, y_clf, test_size=0.2):
    n = len(X)
    n_train = int(n * (1 - test_size))
    return (X[:n_train], y_reg[:n_train], y_clf[:n_train],
            X[n_train:], y_reg[n_train:], y_clf[n_train:])

def train_models(X, y_reg, y_clf, random_state=42) -> Tuple[TrainedModels, Dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, yr_tr, yc_tr, Xte, yr_te, yc_te = time_split(Xs, y_reg, y_clf, test_size=0.2)

    # Regression: next-day return
    reg = GradientBoostingRegressor(random_state=random_state)
    reg.fit(Xtr, yr_tr)
    yr_pred = reg.predict(Xte)

    # Classification: up/down
    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(Xtr, yc_tr)
    yc_proba = clf.predict_proba(Xte)[:, 1]
    yc_pred = (yc_proba >= 0.5).astype(int)

    metrics = {
        "reg_rmse": float(np.sqrt(mean_squared_error(yr_te, yr_pred))),
        "reg_r2": float(r2_score(yr_te, yr_pred)),
        "clf_acc": float(accuracy_score(yc_te, yc_pred)),
        "clf_prec": float(precision_score(yc_te, yc_pred, zero_division=0)),
        "clf_rec": float(recall_score(yc_te, yc_pred, zero_division=0)),
        "clf_auc": float(roc_auc_score(yc_te, yc_proba)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte))
    }

    return TrainedModels(reg, clf, scaler, []), metrics

def save_artifacts(models: TrainedModels, feature_cols, artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(models.regressor, os.path.join(artifacts_dir, "regressor.joblib"))
    joblib.dump(models.classifier, os.path.join(artifacts_dir, "classifier.joblib"))
    joblib.dump(models.scaler,    os.path.join(artifacts_dir, "scaler.joblib"))
    joblib.dump(feature_cols,     os.path.join(artifacts_dir, "feature_cols.joblib"))

def load_artifacts(artifacts_dir="artifacts") -> TrainedModels:
    reg = joblib.load(os.path.join(artifacts_dir, "regressor.joblib"))
    clf = joblib.load(os.path.join(artifacts_dir, "classifier.joblib"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(artifacts_dir, "feature_cols.joblib"))
    tm = TrainedModels(reg, clf, scaler, feature_cols)
    return tm

def predict_next_day(models: TrainedModels, last_row_features: np.ndarray) -> dict:
    Xs = models.scaler.transform(last_row_features.reshape(1, -1))
    next_return_pred = float(models.regressor.predict(Xs)[0])
    up_proba = float(models.classifier.predict_proba(Xs)[0, 1])
    return {"pred_next_return": next_return_pred, "prob_up": up_proba}
