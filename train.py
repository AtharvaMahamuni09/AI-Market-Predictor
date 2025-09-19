\
import argparse
import os
import json
import pandas as pd
import numpy as np
from src.data import download_data, build_features, get_feature_target_matrices
from src.model import train_models, save_artifacts

def main():
    parser = argparse.ArgumentParser(description="Train AI Stock/Crypto Predictor")
    parser.add_argument("--ticker", type=str, required=True, help="e.g., BTC-USD, AAPL")
    parser.add_argument("--start", type=str, default="2017-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    print(f"Downloading {args.ticker} from {args.start} to {args.end or 'today'} ...")
    df = download_data(args.ticker, start=args.start, end=args.end)
    feat_df = build_features(df)
    X, y_reg, y_clf, feature_cols = get_feature_target_matrices(feat_df)

    models, metrics = train_models(X, y_reg, y_clf)
    save_artifacts(models, feature_cols, artifacts_dir=args.artifacts_dir)

    # Save metrics
    os.makedirs(args.artifacts_dir, exist_ok=True)
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("""
================ Training Complete ================
Artifacts saved to: {art}
Metrics:
{met}
""".format(
    art=os.path.abspath(args.artifacts_dir),
    met=json.dumps(metrics, indent=2)
))
