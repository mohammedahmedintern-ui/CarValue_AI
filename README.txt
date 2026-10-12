# CarValue AI — improved dataset version

## Run on Windows
1. Extract this ZIP.
2. Open Command Prompt in the extracted folder.
3. Activate your existing environment if desired:
   `ml_env\Scripts\activate`
4. Install dependencies:
   `python -m pip install -r requirements.txt`
5. Start the app:
   `python -m streamlit run app.py`

The app trains a Random Forest model from `used_car_data.csv` when first opened. It uses car model, manufacturing year, fuel, kilometres driven, city, transmission, and previous owners.

## Important limitation
The supplied dataset has 7,400 Indian used-car listings, including Hyderabad, but the listing dates run from 2019 to May 2021. This is not current market data. Estimates should be treated as a prototype / historical baseline, not as an accurate present-day valuation. Future values are illustrative and assume 10% depreciation per year.
