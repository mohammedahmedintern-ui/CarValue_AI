import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path

st.set_page_config(page_title="CarValue AI", page_icon="🚗", layout="centered")
st.title("🚗 CarValue AI")
st.caption("Used-car price estimate • trained on historical Indian listings from 2019–2021")
DATA_PATH = Path(__file__).parent / "used_car_data.csv"

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    needed = ["car_name","yr_mfr","fuel_type","kms_run","city","transmission","total_owners","sale_price"]
    data = data[needed].copy()
    data["transmission"] = data["transmission"].fillna("Unknown")
    data = data[(data["sale_price"] > 0) & (data["yr_mfr"].between(1990, 2021)) & (data["kms_run"] >= 0)]
    cat = ["car_name","fuel_type","city","transmission"]
    num = ["yr_mfr","kms_run","total_owners"]
    X = data[cat + num]
    y = data["sale_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])
    model = Pipeline([
        ("preprocess", pre),
        ("regressor", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    # Refit on all usable rows for production estimate
    model.fit(X, y)
    return model, data, mae

try:
    model, data, mae = train_model()
except Exception as e:
    st.error(f"Could not load/train the model: {e}")
    st.stop()

st.info("This dataset contains 7,400 listings from multiple Indian cities, dated mainly 2019–2021. Estimates are a historical-data baseline, not live market prices.")
with st.form("valuation"):
    c1, c2 = st.columns(2)
    with c1:
        car_name = st.selectbox("Car model", sorted(data["car_name"].dropna().unique()))
        city = st.selectbox("City", sorted(data["city"].dropna().unique()), index=(sorted(data["city"].dropna().unique()).index("hyderabad") if "hyderabad" in set(data["city"].dropna()) else 0))
        year = st.number_input("Manufacturing year", min_value=1990, max_value=2026, value=2018, step=1)
        kms = st.number_input("Kilometres driven", min_value=0, max_value=1000000, value=50000, step=1000)
    with c2:
        fuels = sorted(data["fuel_type"].dropna().unique())
        fuel = st.selectbox("Fuel type", fuels)
        transmissions = ["manual", "automatic", "Unknown"]
        transmission = st.selectbox("Transmission", transmissions)
        owners = st.selectbox("Number of previous owners", [1,2,3,4,5,6], index=0)
        future_years = st.selectbox("Estimate future value after", [1,2,3,4,5], index=1)
    submitted = st.form_submit_button("Estimate car value", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "car_name": car_name, "yr_mfr": int(year), "fuel_type": fuel,
        "kms_run": int(kms), "city": city, "transmission": transmission,
        "total_owners": int(owners)
    }])
    predicted = max(0, float(model.predict(row)[0]))
    future = predicted * ((1 - 0.10) ** int(future_years))
    st.subheader("Estimated resale value")
    st.metric("Model estimate", f"₹{predicted:,.0f}", help="Machine-learning estimate from the uploaded historical listings.")
    st.metric(f"Illustrative value in {future_years} year(s)", f"₹{future:,.0f}", help="Uses an assumed 10% annual depreciation. Actual depreciation varies.")
    st.caption(f"Validation MAE on a random holdout: approximately ₹{mae:,.0f}. This is an average error indicator, not a guarantee for this individual car.")
    matching = data[(data["car_name"] == car_name) & (data["city"] == city)]
    st.write(f"Historical records for this model in {city.title()}: **{len(matching)}**")
    if len(matching) < 10:
        st.warning("Few matching historical listings are available for this model and city, so the estimate may be less reliable.")
    st.warning("Listings are dated 2019–2021. Use this as a prototype only; compare with current listings and a local dealer before making a purchase or sale.")
