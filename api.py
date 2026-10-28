from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = FastAPI(title="CarValue AI Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "used_car_data.csv"

CATEGORICAL = ["car_name", "fuel_type", "city", "transmission"]
NUMERIC = ["yr_mfr", "kms_run", "total_owners"]
REQUIRED_COLUMNS = CATEGORICAL + NUMERIC + ["sale_price"]


class CarDetails(BaseModel):
    car_name: str
    yr_mfr: int = Field(ge=1990, le=2021)
    fuel_type: str
    kms_run: int = Field(ge=0, le=1_000_000)
    city: str
    transmission: str
    total_owners: int = Field(ge=1, le=6)


@lru_cache(maxsize=1)
def load_data():
    """Read and validate the small source dataset once per process."""
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH.name}")

    data = pd.read_csv(DATA_PATH, usecols=lambda col: col in REQUIRED_COLUMNS)

    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise RuntimeError("Dataset is missing required columns: " + ", ".join(sorted(missing)))

    # Keep only valid rows and avoid retaining unnecessary columns.
    data = data[REQUIRED_COLUMNS].copy()
    data["transmission"] = data["transmission"].fillna("Unknown")
    data = data[
        data["sale_price"].gt(0)
        & data["yr_mfr"].between(1990, 2021)
        & data["kms_run"].ge(0)
    ].copy()

    if data.empty:
        raise RuntimeError("No valid training records were found in the dataset.")

    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip().str.lower()

    return data


@lru_cache(maxsize=1)
def load_model():
    """Train only when a prediction is requested, not when dropdowns load."""
    data = load_data()

    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ])),
        ("regressor", RandomForestRegressor(
            n_estimators=50,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=1,
        )),
    ])

    model.fit(data[CATEGORICAL + NUMERIC], data["sale_price"])
    return model


@app.get("/")
def home():
    return {"status": "CarValue AI API is running"}


@app.get("/health")
def health():
    # Lightweight check: deliberately does not load the dataset or train the model.
    return {"status": "healthy"}


@app.get("/options")
def options():
    # This endpoint must stay lightweight; it must not train the ML model.
    try:
        data = load_data()
        return {
            "models": sorted(data["car_name"].dropna().unique().tolist()),
            "cities": sorted(data["city"].dropna().unique().tolist()),
            "fuels": sorted(data["fuel_type"].dropna().unique().tolist()),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load car options: {exc}") from exc


@app.post("/predict")
def predict(car: CarDetails):
    try:
        model = load_model()
        row = pd.DataFrame([car.model_dump()])
        for col in CATEGORICAL:
            row[col] = row[col].fillna("Unknown").astype(str).str.strip().str.lower()

        prediction = max(0, float(model.predict(row[CATEGORICAL + NUMERIC])[0]))
        data = load_data()
        matching = data[
            data["car_name"].eq(str(car.car_name).strip().lower())
            & data["city"].eq(str(car.city).strip().lower())
        ]

        return {
            "estimated_value": round(prediction),
            "historical_matching_records": int(len(matching)),
            "data_period": "Historical listings, mainly 2019-2021",
            "note": "Prototype estimate, not a live market price.",
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction temporarily unavailable: {exc}") from exc
