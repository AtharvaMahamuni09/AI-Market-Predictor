import os
from datetime import date, timedelta

import streamlit as st

from src.data import (
    download_data,
    build_features,
    get_feature_target_matrices,
)
from src.models import (
    train_and_save,
    load_models,
    predict_next_day,
)

st.title("AI-based Stock & Crypto Predictor")

# ----- Sidebar inputs -----
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")
    start = st.date_input("Start", value=date(2023, 1, 1))
    end = st.date_input("End", value=date.today())
    artifacts_dir = st.text_input("Artifacts dir", value="artifacts")

# ----- Tabs -----
tab1, tab2, tab3 = st.tabs(["📥 Data", "🧠 Train", "🔮 Predict"])

# =========================
# Tab 1: Data
# =========================
with tab1:
    st.subheader("Download & Explore")
    if st.button("Fetch Data"):
        try:
            df = download_data(
                ticker,
                start.isoformat(),
                (end + timedelta(days=1)).isoformat()
            )
            if df is None or len(df) == 0:
                st.warning(f"No data returned for {ticker}. Try a different date range or ticker.")
            else:
                st.session_state["raw_df"] = df
                st.success(f"Loaded {ticker} with {len(df)} rows.")
                st.line_chart(df["Close"])
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    if "raw_df" in st.session_state and st.session_state["raw_df"] is not None and len(st.session_state["raw_df"]) > 0:
        st.dataframe(st.session_state["raw_df"].tail(10))

# =========================
# Tab 2: Train
# =========================
with tab2:
    st.subheader("Train Models")
    if st.button("Build Features & Train"):
        if "raw_df" not in st.session_state:
            st.error("Fetch data first (Data tab).")
        else:
            feat_df = build_features(st.session_state["raw_df"])
            X, y_reg, y_clf, feature_cols = get_feature_target_matrices(feat_df)

            metrics = train_and_save(X, y_reg, y_clf, feature_cols, model_dir=artifacts_dir)
            st.session_state["feature_cols"] = feature_cols
            st.session_state["feat_df"] = feat_df

            st.success(f"Training complete. Artifacts saved to: {artifacts_dir}")
            st.json(metrics)

# =========================
# Tab 3: Predict
# =========================
with tab3:
    st.subheader("Load Artifacts & Predict Next Day")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Artifacts"):
            try:
                reg, clf, features = load_models(model_dir=artifacts_dir)
                st.session_state["reg"] = reg
                st.session_state["clf"] = clf
                st.session_state["features"] = features
                st.success(f"Artifacts loaded from: {artifacts_dir}")
            except Exception as e:
                st.error(f"Error loading models: {e}")

    with col2:
        use_latest = st.checkbox("Use latest fetched data for features", value=True)

    if st.button("Predict Next Day"):
        if not all(k in st.session_state for k in ["reg", "clf", "features"]):
            st.error("Load artifacts first.")
        else:
            # get features from the latest fetched data, or refetch
            if use_latest and "raw_df" in st.session_state:
                feat_df = build_features(st.session_state["raw_df"])
            else:
                df = download_data(
                    ticker,
                    start.isoformat(),
                    (end + timedelta(days=1)).isoformat()
                )
                feat_df = build_features(df)

            # last row as DataFrame so we can select columns by name
            last_row = feat_df.tail(1)

            try:
                pred_return, prob_up = predict_next_day(
                    st.session_state["reg"],
                    st.session_state["clf"],
                    st.session_state["features"],
                    last_row
                )
                st.metric("Predicted next-day return", f"{pred_return:.4%}")
                st.metric("Probability next day UP", f"{prob_up:.2%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # (Optional tiny diagnostic while you’re debugging)
    # st.write("Artifacts dir:", artifacts_dir)
    # if os.path.isdir(artifacts_dir):
    #     st.write("Files:", os.listdir(artifacts_dir))
