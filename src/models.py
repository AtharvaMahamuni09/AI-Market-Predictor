import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, roc_auc_score
)

def train_and_save(X, y_reg, y_clf, feature_cols, model_dir: str = "artifacts"):
    """
    Trains simple models (LinearRegression + LogisticRegression) on full X for demo
    and saves artifacts: reg.pkl, clf.pkl, features.pkl
    Returns a dict of quick-fit metrics (on train set for simplicity).
    """
    os.makedirs(model_dir, exist_ok=True)

    # Regression model
    reg = LinearRegression()
    reg.fit(X, y_reg)

    # Classification model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y_clf)

    # Save artifacts
    with open(os.path.join(model_dir, "reg.pkl"), "wb") as f:
        pickle.dump(reg, f)
    with open(os.path.join(model_dir, "clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, "features.pkl"), "wb") as f:
        pickle.dump(list(feature_cols), f)

    # Quick metrics (train-set; swap to proper split if you prefer)
    y_reg_pred = reg.predict(X)
    reg_rmse = mean_squared_error(y_reg, y_reg_pred, squared=False)
    reg_r2 = r2_score(y_reg, y_reg_pred)

    y_clf_pred = clf.predict(X)
    y_clf_prob = clf.predict_proba(X)[:, 1]
    clf_acc = accuracy_score(y_clf, y_clf_pred)
    clf_prec = precision_score(y_clf, y_clf_pred, zero_division=0)
    clf_rec = recall_score(y_clf, y_clf_pred, zero_division=0)
    clf_auc = roc_auc_score(y_clf, y_clf_prob)

    return {
        "Regression RMSE": float(reg_rmse),
        "Regression R2": float(reg_r2),
        "Classification Accuracy": float(clf_acc),
        "Classification Precision": float(clf_prec),
        "Classification Recall": float(clf_rec),
        "Classification AUC": float(clf_auc),
    }

def load_models(model_dir: str = "artifacts"):
    """
    Loads (reg, clf, features) from model_dir.
    """
    with open(os.path.join(model_dir, "reg.pkl"), "rb") as f:
        reg = pickle.load(f)
    with open(os.path.join(model_dir, "clf.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return reg, clf, features

def predict_next_day(reg, clf, features, latest_row_df):
    """
    latest_row_df: DataFrame that contains the 'features' columns for the most recent day.
    Returns (predicted_next_return, probability_up).
    """
    X_latest = latest_row_df[features].values.reshape(1, -1)
    pred_return = float(reg.predict(X_latest)[0])
    prob_up = float(clf.predict_proba(X_latest)[0, 1])
    return pred_return, prob_up
