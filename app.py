import os
import time
import sqlite3
import hashlib
import secrets
import math

import requests
import numpy as np

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


app = FastAPI(title="StockPilot")

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


DB = "stockpilot.db"
MARKET_CACHE = {}
MARKET_CACHE_TTL = 300


def init_db():
    conn = sqlite3.connect(DB)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            UNIQUE(user_id, ticker)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_price REAL NOT NULL,
            UNIQUE(user_id, ticker)
        )
    """)

    conn.commit()
    conn.close()


init_db()


class AuthRequest(BaseModel):
    username: str
    password: str


class PortfolioRequest(BaseModel):
    ticker: str
    shares: float
    avg_price: float


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_user(token):
    if not token:
        return None

    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT user_id FROM sessions WHERE token=?",
        (token,)
    ).fetchone()
    conn.close()

    return row[0] if row else None


@app.get("/")
def home():
    index = os.path.join(FRONTEND_DIR, "index.html")

    if os.path.exists(index):
        return FileResponse(index)

    return {
        "name": "StockPilot",
        "status": "API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "StockPilot"
    }


@app.post("/api/register")
def register(data: AuthRequest):

    username = data.username.strip().lower()

    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="Username must contain at least 3 characters."
        )

    if len(data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 6 characters."
        )

    conn = sqlite3.connect(DB)

    try:
        conn.execute(
            "INSERT INTO users(username,password) VALUES(?,?)",
            (username, hash_password(data.password))
        )

        conn.commit()

    except sqlite3.IntegrityError:
        conn.close()

        raise HTTPException(
            status_code=400,
            detail="Username already exists."
        )

    conn.close()

    return {
        "message": "Account created successfully."
    }


@app.post("/api/login")
def login(data: AuthRequest):

    username = data.username.strip().lower()

    conn = sqlite3.connect(DB)

    row = conn.execute(
        "SELECT id,password FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if not row:
        conn.close()

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    user_id, password_hash = row

    if password_hash != hash_password(data.password):
        conn.close()

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    token = secrets.token_urlsafe(32)

    conn.execute(
        "INSERT INTO sessions(token,user_id) VALUES(?,?)",
        (token, user_id)
    )

    conn.commit()
    conn.close()

    return {
        "token": token,
        "username": username
    }


def get_market_data(ticker):

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ALPHAVANTAGE_API_KEY is not configured."
        )

    ticker = ticker.upper()

    now = time.time()

    cached = MARKET_CACHE.get(ticker)

    if cached and now - cached["time"] < MARKET_CACHE_TTL:
        return cached["rows"]

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Market data request failed: {str(e)}"
        )

    if "Note" in data:

        raise HTTPException(
            status_code=429,
            detail=data["Note"]
        )

    if "Information" in data:

        raise HTTPException(
            status_code=502,
            detail=data["Information"]
        )

    series = data.get("Time Series (Daily)")

    if not series:

        raise HTTPException(
            status_code=404,
            detail=f"No market data found for {ticker}."
        )

    rows = []

    for date, values in sorted(series.items()):

        try:

            rows.append({
                "date": date,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values["5. volume"])
            })

        except Exception:
            continue

    if len(rows) < 30:

        raise HTTPException(
            status_code=400,
            detail="Not enough historical data for prediction."
        )

    MARKET_CACHE[ticker] = {
        "time": now,
        "rows": rows
    }

    return rows


def build_features(rows):

    closes = np.array(
        [r["close"] for r in rows],
        dtype=float
    )

    features = []
    targets = []

    for i in range(20, len(closes) - 1):

        current = closes[i]

        sma5 = np.mean(closes[i - 5:i])
        sma20 = np.mean(closes[i - 20:i])

        momentum5 = (
            current / closes[i - 5]
        ) - 1

        returns10 = np.diff(
            closes[i - 10:i + 1]
        ) / closes[i - 10:i]

        returns20 = np.diff(
            closes[i - 20:i + 1]
        ) / closes[i - 20:i]

        volatility10 = np.std(returns10)
        volatility20 = np.std(returns20)

        x = [
            current / sma5 - 1,
            current / sma20 - 1,
            momentum5,
            volatility10,
            volatility20
        ]

        next_return = (
            closes[i + 1] / current
        ) - 1

        features.append(x)
        targets.append(next_return)

    return np.array(features), np.array(targets)


def forecast(rows):

    X, y = build_features(rows)

    if len(X) < 20:
        raise HTTPException(
            status_code=400,
            detail="Not enough data for model training."
        )

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = float(
            np.mean(np.abs(predictions - y_test))
        )

        rmse = float(
            math.sqrt(
                np.mean(
                    (predictions - y_test) ** 2
                )
            )
        )

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse
        }

    best_name = min(
        results,
        key=lambda name: results[name]["rmse"]
    )

    best_model = results[best_name]["model"]

    best_model.fit(X, y)

    current_price = rows[-1]["close"]

    latest_X = X[-1].reshape(1, -1)

    predicted_return = float(
        best_model.predict(latest_X)[0]
    )

    target = current_price * (
        1 + predicted_return
    )

    rmse = results[best_name]["rmse"]

    range_size = current_price * (
        rmse * 2
    )

    lower = target - range_size
    upper = target + range_size

    if predicted_return > 0.003:
        direction = "Bullish"

    elif predicted_return < -0.003:
        direction = "Bearish"

    else:
        direction = "Neutral"

    confidence = max(
        0,
        min(
            100,
            100 * (
                1 - min(
                    rmse / 0.05,
                    1
                )
            )
        )
    )

    return {
        "current_price": current_price,
        "predicted_return": predicted_return,
        "target": target,
        "range_low": lower,
        "range_high": upper,
        "direction": direction,
        "confidence": confidence,
        "selected_model": best_name,
        "models": {
            name: {
                "mae": value["mae"],
                "rmse": value["rmse"]
            }
            for name, value in results.items()
        }
    }


@app.get("/api/stock/{ticker}")
def stock(ticker: str):

    rows = get_market_data(ticker)

    prediction = forecast(rows)

    return {
        "ticker": ticker.upper(),
        "history": rows,
        "forecast": prediction
    }


@app.get("/api/me")
def me(authorization: str = Header(default="")):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    conn = sqlite3.connect(DB)

    row = conn.execute(
        "SELECT username FROM users WHERE id=?",
        (user_id,)
    ).fetchone()

    conn.close()

    return {
        "username": row[0] if row else ""
    }


@app.get("/api/watchlist")
def get_watchlist(
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    conn = sqlite3.connect(DB)

    rows = conn.execute(
        "SELECT ticker FROM watchlist WHERE user_id=?",
        (user_id,)
    ).fetchall()

    conn.close()

    return [
        row[0]
        for row in rows
    ]


@app.post("/api/watchlist/{ticker}")
def add_watchlist(
    ticker: str,
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    ticker = ticker.upper()

    conn = sqlite3.connect(DB)

    conn.execute(
        "INSERT OR IGNORE INTO watchlist(user_id,ticker) VALUES(?,?)",
        (user_id, ticker)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Added to watchlist."
    }


@app.delete("/api/watchlist/{ticker}")
def remove_watchlist(
    ticker: str,
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    conn = sqlite3.connect(DB)

    conn.execute(
        "DELETE FROM watchlist WHERE user_id=? AND ticker=?",
        (user_id, ticker.upper())
    )

    conn.commit()
    conn.close()

    return {
        "message": "Removed from watchlist."
    }


@app.get("/api/portfolio")
def get_portfolio(
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    conn = sqlite3.connect(DB)

    rows = conn.execute(
        """
        SELECT ticker, shares, avg_price
        FROM portfolio
        WHERE user_id=?
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    return [
        {
            "ticker": row[0],
            "shares": row[1],
            "avg_price": row[2]
        }
        for row in rows
    ]


@app.post("/api/portfolio")
def add_portfolio(
    data: PortfolioRequest,
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    if data.shares <= 0:
        raise HTTPException(
            status_code=400,
            detail="Shares must be greater than zero."
        )

    if data.avg_price <= 0:
        raise HTTPException(
            status_code=400,
            detail="Average price must be greater than zero."
        )

    ticker = data.ticker.upper()

    conn = sqlite3.connect(DB)

    conn.execute(
        """
        INSERT INTO portfolio(
            user_id,
            ticker,
            shares,
            avg_price
        )
        VALUES(?,?,?,?)

        ON CONFLICT(user_id,ticker)
        DO UPDATE SET
            shares=excluded.shares,
            avg_price=excluded.avg_price
        """,
        (
            user_id,
            ticker,
            data.shares,
            data.avg_price
        )
    )

    conn.commit()
    conn.close()

    return {
        "message": "Portfolio updated."
    }


@app.delete("/api/portfolio/{ticker}")
def remove_portfolio(
    ticker: str,
    authorization: str = Header(default="")
):

    user_id = get_user(authorization)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated."
        )

    conn = sqlite3.connect(DB)

    conn.execute(
        """
        DELETE FROM portfolio
        WHERE user_id=? AND ticker=?
        """,
        (
            user_id,
            ticker.upper()
        )
    )

    conn.commit()
    conn.close()

    return {
        "message": "Holding removed."
    }
