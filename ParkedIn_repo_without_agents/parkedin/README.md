# ParkedIn: demand-based dynamic pricing for urban parking

A system that replays real parking data as if it were live. Every few seconds it processes one
30-minute slot: it checks its last forecasts, predicts the next 3 hours, labels each car park's
demand, sets a new price for every car park, and saves the results for the dashboard.

| Step | File | Method | What it does |
| --- | --- | --- | --- |
| Data preparation | `data.py` | pandas | Cleans the CSV, aligns readings to 30-minute slots, builds features, splits the days |
| Forecasting | `forecasting.py` | XGBoost (one model per step ahead) | Predicts each lot's occupancy for the next 6 slots (3 hours) |
| Demand levels | `demand_levels.py` | K-Means, 4 clusters | Labels each lot's demand: low, medium, high, peak |
| Pricing | `pricing.py` | Rule-based start + grid search | Picks each lot's next price inside its min/max limits |
| Monitoring | `monitoring.py` | Rolling forecast error | Raises alerts and retrains the forecast models on drift |

`simulation.py` runs these steps for every slot, `db.py` is the SQLite database the
simulation writes and the dashboard reads, and `dashboard/` is the Streamlit app (live map,
lot details, admin, AI assistant).

## 1. One-time setup

You need **Python 3.10 to 3.13** and an internet connection (for installing packages and
for the map background).

```bash
# from inside the parkedin folder
python -m venv .venv

# activate it
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat
# macOS / Linux:          source .venv/bin/activate

pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. Get the dataset

Download `dataset.csv` (14 lots, 18,368 rows) and save it as `data/raw/dataset.csv`:

- Open https://github.com/sh-kartik18/Dynamic-Pricing-for-Urban-Parking-Lots/blob/main/dataset.csv
- Click the download button (Download raw file), then move the file into `data/raw/`.

Or from a terminal (macOS / Linux / Git Bash):

```bash
curl -L -o data/raw/dataset.csv https://raw.githubusercontent.com/sh-kartik18/Dynamic-Pricing-for-Urban-Parking-Lots/main/dataset.csv
```

## 3. Train the models (about 1 to 3 minutes)

```bash
python train.py
python evaluate_pricing.py
```

`train.py` builds the features, compares naive, ARIMA, Random Forest and XGBoost forecasts,
trains K-Means, and saves everything in `models/`. `evaluate_pricing.py` compares fixed
price, rules only, and rules + grid search over the whole test period. Both print tables you
can copy into the report; the Admin page shows them too.

## 4. Run the system (two terminals)

Activate the virtual environment in both terminals, in the `parkedin` folder.

```bash
# Terminal 1: the simulation
python simulation.py

# Terminal 2: the dashboard (opens http://localhost:8501 in your browser)
streamlit run dashboard/Home.py
```

Press **Start simulation** on the Home page. Prices update every 5 seconds. Use the
**Admin** page to change the speed, the pricing balance, each lot's price limits, or to
reset the replay. Stop everything with `Ctrl + C` in both terminals.

## 5. Things to know

- **Start the simulation first.** It creates `parkedin.db`. To start completely fresh,
  stop both programs and delete `parkedin.db` (plus `parkedin.db-wal` / `-shm` if present).
- **After changing `config.yaml`**, restart the simulation. Price weights and speed can
  also be changed live on the Admin page.
- **After changing any training code**, run `python train.py` again.
- **Map not showing?** The map background needs internet. Everything else works offline.
- **Assumption:** the data never shows how drivers react to price, so the pricing and
  the revenue comparison use an assumed price sensitivity (`elasticity: 0.4` in
  `config.yaml`). `evaluate_pricing.py` also tests 0.2 and 0.6.
- **Data fixes made during data preparation:** readings are aligned to 30-minute slots, a few
  missing readings are interpolated, and one lot (`BHMMBMMBX01`) whose coordinates were far
  from the others is placed next to them on the map.

## 6. Put the dashboard online (Streamlit Community Cloud)

1. Push the repository to GitHub. Keep `requirements.txt` in the **repository root** (next to
   the `parkedin` folder); Streamlit Cloud only looks there or next to the main file.
2. On share.streamlit.io, create an app with main file path `parkedin/dashboard/Home.py`.
3. On the first visit the dashboard sets itself up: it downloads the dataset, trains the
   models (about 2 to 3 minutes) and starts the simulation in the background. Then press
   **Start simulation**. No second terminal is needed online.

## 7. AI assistant (optional)

The **AI Assistant** page answers questions about the parking data in plain English: the best
car park near you, why a price is what it is, a weekly report, questions about the data,
unusual patterns, and revenue ideas. It sends a short summary of the database to Gemini and
asks for the answer in a fixed JSON format; if Gemini fails or returns a broken answer, the
same question goes to OpenAI. It only reads summaries: it cannot run SQL or change prices.

Add one or both keys as Streamlit secrets (`.streamlit/secrets.toml` on a laptop, or the
app's Secrets on Streamlit Community Cloud) or as environment variables:

```toml
GEMINI_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
```

Optional settings: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `OPENAI_MODEL` (default
`gpt-4.1-mini`, which OpenAI retires on 14 January 2027) and `AI_PRIMARY_PROVIDER`
(`openai` to try OpenAI first). Without a key, everything except this page works normally.

## Results on the test days (11 days, 196 slots)

Forecast error (MAE of occupancy rate, lower is better):

| Slots ahead | Naive | ARIMA | Random Forest | XGBoost |
| --- | --- | --- | --- | --- |
| 1 | 0.0425 | 0.0450 | 0.0166 | **0.0156** |
| 3 | 0.1111 | 0.0961 | 0.0358 | **0.0350** |
| 6 | 0.1779 | 0.1330 | 0.0550 | **0.0538** |

Pricing at the assumed sensitivity of 0.4:

| Strategy | Revenue vs fixed Rs 40 | Avg distance from 85% full | Lot-slots 95%+ full | Avg price |
| --- | --- | --- | --- | --- |
| Fixed price | 0% | 0.346 | 137 | Rs 40.0 |
| Rules only | -9.3% | 0.293 | 0 | Rs 32.6 |
| Rules + grid search | **+2.3%** | 0.316 | **0** | Rs 42.1 |

## Project structure

```
parkedin/
├── data.py            data cleaning, features, train/validation/test split
├── forecasting.py     XGBoost forecasts for the next 6 slots
├── demand_levels.py   K-Means demand levels
├── pricing.py         rule-based start + grid search pricing
├── monitoring.py      forecast error, drift alerts, retraining
├── dashboard/         Streamlit app (Home.py + pages/)
├── data/raw/          put dataset.csv here
├── data/processed/    created by train.py
├── models/            created by train.py
├── common.py          paths and config loading
├── config.yaml        all settings
├── db.py              SQLite helpers (the shared "blackboard")
├── simulation.py      runs every step for each replayed slot
├── train.py           trains and compares models
├── evaluate_pricing.py  compares pricing strategies
├── gemini_assistant.py  AI assistant (Gemini, OpenAI as backup)
└── requirements.txt
```
