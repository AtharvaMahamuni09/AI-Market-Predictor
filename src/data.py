import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import io, contextlib, warnings, logging

# Quiet common noise from yfinance and urllib3
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

def _silently(func, *args, **kwargs):
    """Run a function while suppressing noisy stdout/stderr prints."""
    f_out, f_err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
            return func(*args, **kwargs)
# --- Helpers ---------------------------------------------------------------

def _session():
    # Use a browser-like User-Agent to avoid CF/HTML challenges
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    })
    return s

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # Some paths give "Adj Close" or "Adj close"
    cols = {c: c.title() for c in df.columns}
    df = df.rename(columns=cols)
    # keep only standard OHLCV if available
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].copy()
    # ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna()
    df = df.sort_index()
    return df

# --- Download strategies ---------------------------------------------------

def _try_yf_download(ticker, start, end, sess):
    return _silently(
        yf.download,
        ticker, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False,
        threads=False, session=sess
    )

def _try_yf_history(ticker, start, end, sess):
    t = yf.Ticker(ticker, session=sess)
    return _silently(
        t.history,
        start=start, end=end,
        interval="1d", auto_adjust=True
    )

def _try_yf_period(ticker, period, sess):
    return _silently(
        yf.download,
        ticker, period=period, interval="1d",
        auto_adjust=True, progress=False, threads=False,
        session=sess
    )

def _try_stooq(ticker, start, end):
    # Final fallback via pandas-datareader (stooq). Dates are inclusive; stooq returns most recent first.
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, "stooq", start=start, end=end)
        df = df[::-1]  # ascending
        return df
    except Exception:
        return pd.DataFrame()

# --- Public API ------------------------------------------------------------

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    sess = _session()
    attempts = [
        ("yf.download", lambda: _try_yf_download(ticker, start, end, sess)),
        ("ticker.history", lambda: _try_yf_history(ticker, start, end, sess)),
        ("yf.download(period=5y)", lambda: _try_yf_period(ticker, "5y", sess)),
        ("stooq", lambda: _try_stooq(ticker, start, end)),
    ]

    last_err = None
    for name, fn in attempts:
        try:
            df = fn()
            df = _standardize(df)
            if len(df) > 0:
                return df
        except Exception as e:
            last_err = e
        time.sleep(0.8)  # brief backoff

    raise RuntimeError(f"Failed to fetch data for {ticker}. Last error: {last_err}")

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = np.log1p(out["Return"])
    out["Volatility20"] = out["Return"].rolling(20).std()
    out["SMA5"] = out["Close"].rolling(5).mean()
    out["SMA10"] = out["Close"].rolling(10).mean()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["EMA12"] = _ema(out["Close"], 12)
    out["EMA26"] = _ema(out["Close"], 26)
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACDSignal"] = _ema(out["MACD"], 9)
    out["RSI14"] = _rsi(out["Close"], 14)
    for lag in [1, 2, 3, 5, 10]:
        out[f"RetLag{lag}"] = out["Return"].shift(lag)
    out["NextReturn"] = out["Return"].shift(-1)
    out["NextUp"] = (out["NextReturn"] > 0).astype(int)
    out = out.dropna()
    return out

def get_feature_target_matrices(feat_df: pd.DataFrame):
    feature_cols = [
        "Volatility20","SMA5","SMA10","SMA20","EMA12","EMA26",
        "MACD","MACDSignal","RSI14","LogReturn","RetLag1","RetLag2","RetLag3","RetLag5","RetLag10"
    ]
    X = feat_df[feature_cols].values
    y_reg = feat_df["NextReturn"].values
    y_clf = feat_df["NextUp"].values
    return X, y_reg, y_clf, feature_cols
