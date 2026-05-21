"""Build correlation matrices for NSE sectoral index stocks using Kite Connect.

For each configured NSE sectoral index (Nifty Pharma, Nifty Auto, ...) the
script:
  1. Auto-discovers the index constituents (downloads NSE's published CSV,
     falling back to a hardcoded list if the download fails).
  2. Fetches historical close prices for each stock via Kite Connect, in
     chunks that respect the API's per-request date-range limits. Daily and
     intraday intervals are both supported.
  3. Computes a correlation matrix from returns (not raw price levels).
  4. Saves the matrix as a CSV and a heatmap PNG (one pair per index).

Credentials are read from a `.env` file. The Kite access token expires daily,
so the script validates the cached token and runs an interactive login only
when a refresh is needed, then writes the fresh token back to `.env`.
"""

import io
import os
import sys
import time
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; we save PNGs instead of showing
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv, set_key

try:
    from kiteconnect import KiteConnect
except ImportError:
    sys.exit("kiteconnect is not installed. Run: pip install -r requirements.txt")


# --------------------------------------------------------------------------
# Configuration  --  edit these
# --------------------------------------------------------------------------

ENV_PATH = ".env"
OUTPUT_DIR = "output"

# Date range for historical data (format: YYYY-MM-DD).
START_DATE = "2024-05-21"
END_DATE = "2025-05-21"

# Candle interval. One of Kite Connect's supported values:
#   minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day
INTERVAL = "day"

RATE_LIMIT_SLEEP = 0.35  # seconds between historical-data calls (~3 req/s)

# Kite caps the date span of a single historical-data request, and the cap
# depends on the interval. Wider ranges are fetched in chunks no larger than
# the value below. (Daily candles are already within trading hours, so no
# trading-hours filtering is needed.)
MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}

# A browser-like User-Agent: niftyindices.com rejects bare programmatic clients.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
}

# Each index: the official constituent CSV URL plus a hardcoded fallback list
# (tradingsymbols) used when the download is blocked or fails. Add more NSE
# sectoral indexes here by following the same shape.
INDICES = {
    "Nifty Pharma": {
        "csv_url": "https://niftyindices.com/IndexConstituent/ind_niftypharmalist.csv",
        "fallback_symbols": [
            "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "TORNTPHARM",
            "ZYDUSLIFE", "LUPIN", "AUROPHARMA", "ALKEM", "MANKIND",
            "GLENMARK", "BIOCON", "LAURUSLABS", "GRANULES", "NATCOPHARM",
            "AJANTPHARM", "IPCALAB", "ABBOTINDIA", "PPLPHARMA", "GLAND",
        ],
    },
    "Nifty Auto": {
        "csv_url": "https://niftyindices.com/IndexConstituent/ind_niftyautolist.csv",
        "fallback_symbols": [
            "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT",
            "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY", "BOSCHLTD", "BHARATFORG",
            "MOTHERSON", "BALKRISIND", "MRF", "EXIDEIND", "TIINDIA",
        ],
    },
}


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------

def get_kite():
    """Return an authenticated KiteConnect client.

    API key/secret are permanent; the access token expires every day. A cached
    token is reused if still valid, otherwise an interactive login is run and
    the new token is persisted back to `.env`.
    """
    load_dotenv(ENV_PATH)
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not api_secret:
        sys.exit(
            "Missing KITE_API_KEY / KITE_API_SECRET. "
            "Copy .env.example to .env and fill them in."
        )

    kite = KiteConnect(api_key=api_key)

    # Try the cached token first.
    if access_token:
        kite.set_access_token(access_token)
        try:
            kite.profile()
            print("Using cached Kite access token.")
            return kite
        except Exception:
            print("Cached access token is invalid or expired; refreshing.")

    # Interactive login to mint a fresh token for today.
    print("\nOpen this URL in a browser and log in to Zerodha:")
    print("  " + kite.login_url())
    print(
        "After login you are redirected to your app's redirect URL. Copy the "
        "`request_token` query parameter from that URL."
    )
    request_token = input("Paste request_token here: ").strip()

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    kite.set_access_token(access_token)

    # Persist so the rest of today's runs skip the login step.
    set_key(ENV_PATH, "KITE_ACCESS_TOKEN", access_token)
    print("New access token saved to .env\n")
    return kite


# --------------------------------------------------------------------------
# Index constituents
# --------------------------------------------------------------------------

def fetch_constituents(index_name, cfg):
    """Return the list of tradingsymbols for an index.

    Downloads the official NSE constituent CSV; on any failure logs a warning
    and returns the hardcoded fallback list.
    """
    try:
        resp = requests.get(cfg["csv_url"], headers=HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = [str(s).strip() for s in df["Symbol"].dropna()]
        if not symbols:
            raise ValueError("CSV contained no symbols")
        print(f"{index_name}: fetched {len(symbols)} constituents from NSE.")
        return symbols
    except Exception as exc:
        symbols = cfg["fallback_symbols"]
        print(
            f"{index_name}: could not download constituents ({exc}); "
            f"using hardcoded fallback list of {len(symbols)} stocks."
        )
        return symbols


# --------------------------------------------------------------------------
# Instrument tokens & price history
# --------------------------------------------------------------------------

def build_token_map(kite):
    """Map NSE tradingsymbol -> instrument_token (needed for historical data).

    Fetched once for the whole run so per-symbol lookups cost no API calls.
    """
    instruments = kite.instruments("NSE")
    return {inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments}


def fetch_close_series(kite, token, symbol, from_date, to_date):
    """Return a pandas Series of close prices for one instrument.

    Kite limits the date span of a single historical-data request, so wide
    ranges are split into chunks no larger than MAX_DAYS_PER_REQUEST[INTERVAL].
    """
    max_days = MAX_DAYS_PER_REQUEST[INTERVAL]
    chunks = []
    current = from_date
    while current <= to_date:
        chunk_end = min(current + timedelta(days=max_days - 1), to_date)
        candles = kite.historical_data(
            token,
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
            INTERVAL,
        )
        if candles:
            frame = pd.DataFrame(candles)
            chunks.append(frame.set_index("date")["close"])
        current = chunk_end + timedelta(days=1)
        time.sleep(RATE_LIMIT_SLEEP)  # respect Kite's rate limit

    if not chunks:
        return None
    series = pd.concat(chunks)
    series = series[~series.index.duplicated(keep="last")]  # drop chunk overlaps
    series.sort_index(inplace=True)
    series.name = symbol
    return series


def collect_close_prices(kite, symbols, token_map, from_date, to_date):
    """Fetch close-price series for every symbol and return them as a DataFrame."""
    series_list = []
    for symbol in symbols:
        token = token_map.get(symbol)
        if token is None:
            print(f"  skip {symbol}: not found in NSE instrument list.")
            continue
        try:
            series = fetch_close_series(kite, token, symbol, from_date, to_date)
            if series is None or series.empty:
                print(f"  skip {symbol}: no historical data returned.")
                continue
            series_list.append(series)
            print(f"  fetched {symbol}: {len(series)} candles.")
        except Exception as exc:
            print(f"  skip {symbol}: historical-data error ({exc}).")

    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)


# --------------------------------------------------------------------------
# Correlation & plotting
# --------------------------------------------------------------------------

def build_correlation(close_df):
    """Correlation matrix of returns.

    Returns (not raw price levels) are used because price series are
    non-stationary and trending, which inflates apparent correlation.
    """
    returns = close_df.pct_change().dropna(how="all")
    return returns.corr()


def plot_heatmap(corr, title, out_path):
    """Save a labelled correlation heatmap to out_path."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.6)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    # Annotate each cell with its correlation value.
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_dates():
    """Validate and parse the configured date range."""
    try:
        start = datetime.strptime(START_DATE, "%Y-%m-%d").date()
        end = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    except ValueError:
        sys.exit("START_DATE / END_DATE must be in YYYY-MM-DD format.")
    if start >= end:
        sys.exit("START_DATE must be earlier than END_DATE.")
    return start, end


def main():
    if INTERVAL not in MAX_DAYS_PER_REQUEST:
        sys.exit(f"INTERVAL must be one of: {', '.join(MAX_DAYS_PER_REQUEST)}")
    from_date, to_date = parse_dates()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    kite = get_kite()
    print("Loading NSE instrument list...")
    token_map = build_token_map(kite)

    for index_name, cfg in INDICES.items():
        print(f"\n=== {index_name} ===")
        symbols = fetch_constituents(index_name, cfg)

        close_df = collect_close_prices(kite, symbols, token_map, from_date, to_date)
        if close_df.empty or close_df.shape[1] < 2:
            print(f"{index_name}: not enough data to build a correlation matrix.")
            continue

        corr = build_correlation(close_df)
        if len(close_df) < 100:
            print(
                f"  warning: only {len(close_df)} data points - correlation "
                "may be unreliable; widen the date range or use a finer interval."
            )

        slug = index_name.replace(" ", "_")
        csv_path = os.path.join(OUTPUT_DIR, f"{slug}_correlation_{INTERVAL}.csv")
        png_path = os.path.join(
            OUTPUT_DIR, f"{slug}_correlation_{INTERVAL}_heatmap.png"
        )

        corr.to_csv(csv_path)
        plot_heatmap(
            corr,
            f"{index_name} - Return Correlation "
            f"({INTERVAL}, {START_DATE} to {END_DATE})",
            png_path,
        )

        print(f"{index_name}: {corr.shape[0]} stocks correlated.")
        print(f"  saved {csv_path}")
        print(f"  saved {png_path}")


if __name__ == "__main__":
    main()
