import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# LOAD MODEL
model = joblib.load(r"D:\linera Regression Model\house_price_xgb_model.pkl")

# TITLE
st.title("🏠 House Price Forecast\n(Next 5-Year)")

# INPUTS
area = st.number_input("Area", 500, 10000, 1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
location = st.number_input("Location Code", 10, 100, 50)
current_year = st.number_input("Current Year", 2025, 2035, 2026)

# BUTTON
if st.button("Predict Future Prices"):

    # CURRENT PRICE
    base_price = model.predict([
        [area, bedrooms, bathrooms, location]
    ])[0]

    years = []
    prices = []

    growth_rate = 0.08  # 8%

    for i in range(1, 6):

        future_year = current_year + i

        future_price = base_price * ((1 + growth_rate) ** i)

        years.append(future_year)
        prices.append(future_price)

    # TABLE
    result = pd.DataFrame({
        "Year": years,
        "Predicted Price": prices
    })

    st.dataframe(result)

    # CHART
    fig, ax = plt.subplots()

    ax.plot(years, prices, marker='o')

    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Next 5-Year Forecast")

    st.pyplot(fig)
