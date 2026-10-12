from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from functools import lru_cache
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


app = FastAPI(title="CarValue AI Prediction API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_PATH = Path(__file__).parent / "used_car_data.csv"

class CarDetails(BaseModel):
    car_name: str
    yr_mfr: int = Field(ge=1990, le=2021)
    fuel_type: str
    kms_run: int = Field(ge=0, le=1000000)
    city: str
    transmission: str
    total_owners: int = Field(ge=1, le=6)

@lru_cache(maxsize=1)
def load_model():
    data = pd.read_csv(DATA_PATH)
    cols = [
        "car_name", "yr_mfr", "fuel_type", "kms_run",
        "city", "transmission", "total_owners", "sale_price"
    ]
    data = data[cols].copy()
    data["transmission"] = data["transmission"].fillna("Unknown")
    data = data[
        (data["sale_price"] > 0)
        & data["yr_mfr"].between(1990, 2021)
        & (data["kms_run"] >= 0)
    ]

    cat = ["car_name", "fuel_type", "city", "transmission"]
    num = ["yr_mfr", "kms_run", "total_owners"]

    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num)
        ])),
        ("regressor", RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(data[cat + num], data["sale_price"])
    return model, data

@app.get("/")
def home():
    return {"status": "CarValue AI API is running"}

@app.get("/options")
def options():
    model, data = load_model()
    return {
        "models": sorted(data["car_name"].dropna().unique().tolist()),
        "cities": sorted(data["city"].dropna().unique().tolist()),
        "fuels": sorted(data["fuel_type"].dropna().unique().tolist())
    }

@app.post("/predict")
def predict(car: CarDetails):
    model, data = load_model()
    row = pd.DataFrame([car.model_dump()])
    prediction = max(0, float(model.predict(row)[0]))
    matching = data[
        (data["car_name"] == car.car_name)
        & (data["city"] == car.city)
    ]
    return {
        "estimated_value": round(prediction),
        "historical_matching_records": int(len(matching)),
        "data_period": "Historical listings, mainly 2019-2021",
        "note": "Prototype estimate, not a live market price."
    }