import os, sqlite3, hashlib, secrets, math
from datetime import datetime import time timedelta
from typing import Optional
import requests, numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DB="stockpilot.db"; KEY=os.getenv("ALPHAVANTAGE_API_KEY","")
app=FastAPI(title="StockPilot Working App",version="4.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT UNIQUE,password TEXT);
    CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,user_id INTEGER);
    CREATE TABLE IF NOT EXISTS watchlist(user_id INTEGER,ticker TEXT,UNIQUE(user_id,ticker));
    CREATE TABLE IF NOT EXISTS portfolio(user_id INTEGER,ticker TEXT,shares REAL,avg_price REAL,UNIQUE(user_id,ticker));"""); c.commit(); return c

def pw(x): return hashlib.sha256(x.encode()).hexdigest()
def user(token):
    c=db(); r=c.execute("SELECT user_id FROM sessions WHERE token=?",(token or "",)).fetchone()
    if not r: raise HTTPException(401,"Please log in.")
    return r["user_id"]

class Auth(BaseModel): username:str; password:str
class Holding(BaseModel): ticker:str; shares:float; avg_price:float

@app.post("/api/register")
def register(a:Auth):
    if len(a.username)<3 or len(a.password)<6: raise HTTPException(400,"Username must be 3+ chars and password 6+ chars.")
    c=db()
    try: c.execute("INSERT INTO users(username,password) VALUES(?,?)",(a.username.lower(),pw(a.password))); c.commit()
    except sqlite3.IntegrityError: raise HTTPException(409,"Username already exists.")
    return {"ok":True}

@app.post("/api/login")
def login(a:Auth):
    c=db(); r=c.execute("SELECT id FROM users WHERE username=? AND password=?",(a.username.lower(),pw(a.password))).fetchone()
    if not r: raise HTTPException(401,"Invalid username or password.")
    token=secrets.token_urlsafe(32); c.execute("INSERT INTO sessions VALUES(?,?)",(token,r["id"])); c.commit()
    return {"token":token,"username":a.username.lower()}

def market(t):
    if not KEY: raise HTTPException(503,"Add ALPHAVANTAGE_API_KEY to enable live market data.")
    r=requests.get("https://www.alphavantage.co/query",params={"function":"TIME_SERIES_DAILY","symbol":t,"outputsize":"compact","apikey":KEY},timeout=20)
    j=r.json(); s=j.get("Time Series (Daily)")
    if not s: raise HTTPException(502,j.get("Note") or j.get("Information") or "Market data unavailable.")
    rows=sorted([(d,float(v["4. close"])) for d,v in s.items()])
    MARKET_CACHE[t] = {"time": now, "rows": rows}
    return rows

def forecast(rows):
    p=np.array([x[1] for x in rows],float); X=[]; y=[]
    for i in range(30,len(p)-1):
        ret=np.diff(p[:i+1])/p[:i]
        X.append([p[i]/np.mean(p[i-4:i+1])-1,p[i]/np.mean(p[i-19:i+1])-1,p[i]/p[i-6]-1,
                  np.std(ret[-10:]),np.std(ret[-20:])])
        y.append(p[i+1]/p[i]-1)
    X=np.array(X); y=np.array(y); cut=max(40,int(len(X)*.8))
    rf=RandomForestRegressor(n_estimators=120,max_depth=7,min_samples_leaf=3,random_state=42).fit(X[:cut],y[:cut])
    gb=GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=.04,random_state=42).fit(X[:cut],y[:cut])
    pr1=rf.predict(X[cut:]); pr2=gb.predict(X[cut:])
    rm1=float(np.sqrt(np.mean((pr1-y[cut:])**2))); rm2=float(np.sqrt(np.mean((pr2-y[cut:])**2)))
    model=rf if rm1<=rm2 else gb; name="Random Forest" if model is rf else "Gradient Boosting"
    model.fit(X,y); pred=float(model.predict(X[-1:])[0]); cur=p[-1]; target=cur*(1+pred)
    rm=min(rm1,rm2); band=max(.012,min(.10,rm*2.2))
    return {"direction":"Bullish" if pred>.003 else ("Bearish" if pred<-.003 else "Neutral"),
            "return_pct":round(pred*100,2),"target":round(target,2),"low":round(target*(1-band),2),
            "high":round(target*(1+band),2),"confidence":max(50,min(86,int(72-min(rm*900,20)))),
            "model":name,"rmse":round(rm,5)}

@app.get("/api/health")
def health(): return {"status":"ok"}

@app.get("/api/stock/{ticker}")
def stock(ticker:str):
    t=ticker.upper(); rows=market(t); f=forecast(rows)
    return {"ticker":t,"price":rows[-1][1],"updated":rows[-1][0],"forecast":f,
            "history":[{"date":d,"price":p} for d,p in rows[-120:]]}

@app.get("/api/me")
def me(authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db(); r=c.execute("SELECT username FROM users WHERE id=?",(uid,)).fetchone()
    return {"username":r["username"]}

@app.get("/api/watchlist")
def watch(authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db()
    return [r["ticker"] for r in c.execute("SELECT ticker FROM watchlist WHERE user_id=? ORDER BY ticker",(uid,)).fetchall()]

@app.post("/api/watchlist/{ticker}")
def add_watch(ticker:str,authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db(); c.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)",(uid,ticker.upper())); c.commit(); return {"ok":True}

@app.delete("/api/watchlist/{ticker}")
def del_watch(ticker:str,authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db(); c.execute("DELETE FROM watchlist WHERE user_id=? AND ticker=?",(uid,ticker.upper())); c.commit(); return {"ok":True}

@app.get("/api/portfolio")
def portfolio(authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db(); return [dict(r) for r in c.execute("SELECT ticker,shares,avg_price FROM portfolio WHERE user_id=?",(uid,)).fetchall()]

@app.post("/api/portfolio")
def setholding(h:Holding,authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db()
    c.execute("""INSERT INTO portfolio VALUES(?,?,?,?) ON CONFLICT(user_id,ticker) DO UPDATE SET shares=excluded.shares,avg_price=excluded.avg_price""",(uid,h.ticker.upper(),h.shares,h.avg_price))
    c.commit(); return {"ok":True}

@app.delete("/api/portfolio/{ticker}")
def delholding(ticker:str,authorization:Optional[str]=Header(None)):
    uid=user(authorization); c=db(); c.execute("DELETE FROM portfolio WHERE user_id=? AND ticker=?",(uid,ticker.upper())); c.commit(); return {"ok":True}
