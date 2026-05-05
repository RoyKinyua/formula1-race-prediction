# F1 Race Prediction Platform

Predicts F1 race winners and podium finishes using machine learning. Data covers 2021–2025 across 114 races.

---

## Data Sources

- **Jolpica (Ergast)** — race results, qualifying, pit stops, standings
- **FastF1** — lap times, sector times, tyre compounds, weather
- **OpenF1** — tyre stints, safety car and race control events

## Seasons

| Season | Races |
|--------|-------|
| 2021   | 22    |
| 2022   | 22    |
| 2023   | 22    |
| 2024   | 24    |
| 2025   | 24*   |

*Up to the last completed round.

---

## Models

Trained on 2021–2024, tested on 2025.

| Model | Target | Accuracy | AUC |
|-------|--------|----------|-----|
| XGBoost | Race winner | 99.3% | 1.000 |
| XGBoost | Podium finish | 98.3% | 0.997 |
| Random Forest | Race winner | 97.7% | 0.998 |
| Logistic Regression | Race winner | 97.5% | 0.994 |

---

## Setup

```bash
pip install uv
uv sync
```

Create `config/settings.yaml` with your PostgreSQL credentials:

```yaml
database:
  host:     your-host
  port:     5432
  name:     your-db
  user:     your-user
  password: your-password
  sslmode:  require
```

Run the schema in your PostgreSQL client:
```
sql/schema_postgres.sql
```

---

## Running the Pipeline

```bash
# 1. Ingest
python -m src.ingestion.ingest_ergast --season 2024
python -m src.ingestion.ingest_fastf1 --season 2024
python -m src.ingestion.ingest_openf1 --year 2024

# 2. Clean
python -m src.processing.clean_data --season 2024 --save-csv

# 3. Build features
python -c "from src.features.build_features import build_features; build_features([2021,2022,2023,2024,2025])"

# 4. Train
python -c "from src.models.train import train; train([2021,2022,2023,2024],[2025])"

# 5. Run app
set PYTHONPATH=. && streamlit run app/streamlit_app.py
```

---

## Tech Stack

- Python 3.11 
- uv 
- PostgreSQL 
- XGBoost 
- scikit-learn 
- FastF1
- OpenF1
- Ergast
- Streamlit 
- Plotly
