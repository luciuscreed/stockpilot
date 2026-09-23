# StockPilot — online deployment package

This version serves the web interface and API from the same FastAPI service, so
you only need one web deployment.

## Fastest route to an online app

1. Create a GitHub repository.
2. Upload the contents of this folder to the repository.
3. Create a Render Web Service from that repository.
4. Set **Root Directory** to `backend`.
5. Build command:
   `pip install -r requirements.txt`
6. Start command:
   `uvicorn app:app --host 0.0.0.0 --port $PORT`
7. Add the secret environment variable:
   `ALPHAVANTAGE_API_KEY=YOUR_KEY`
8. Deploy.

Render provides a public `onrender.com` URL for the service and HTTPS for public
web services. The included `render.yaml` can also be used as the deployment
configuration.

## Local use

From `backend`:
```powershell
python -m pip install -r requirements.txt
$env:ALPHAVANTAGE_API_KEY="YOUR_KEY"
python -m uvicorn app:app --reload
```
Then open:
`http://127.0.0.1:8000/`

## Important

The app is an analysis/forecasting prototype. It does not place trades and its
forecasts are estimates, not guarantees. Before real-money use, upgrade password
hashing, sessions, database, rate limiting, market-data licensing, validation,
monitoring, backups, and applicable financial/regulatory controls.
