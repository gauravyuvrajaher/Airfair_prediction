import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(page_title="✈️ Airfare Prediction App", layout="wide")

# -------------------------------------------------------
# Sidebar – Model Settings
# -------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

test_size = st.sidebar.slider("Test Size (%)", 20, 40, 30) / 100

# -------------------------------------------------------
# Load & Prepare Data (GitHub + Local Safe)
# -------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel(
        "https://raw.githubusercontent.com/gauravyuvrajaher/Airfair_prediction/main/Airfares_data.xlsx"
    )

    # Encode categorical variables
    df["VACATION"] = df["VACATION"].map({"Yes": 1, "No": 0})
    df["SW"] = df["SW"].map({"Yes": 1, "No": 0})
    df["SLOT"] = df["SLOT"].map({"Controlled": 1, "Free": 0})
    df["GATE"] = df["GATE"].map({"Controlled": 1, "Free": 0})

    features = [
        "COUPON", "NEW", "VACATION", "SW", "HI",
        "S_INCOME", "E_INCOME", "S_POP", "E_POP",
        "SLOT", "GATE", "PAX", "DISTANCE"
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, "FARE"]

    return X, y

X, y = load_and_prepare_data()

# -------------------------------------------------------
# Train / Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -------------------------------------------------------
# Model Training
# -------------------------------------------------------
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

else:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

model.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# -------------------------------------------------------
# UI – Title & Metrics
# -------------------------------------------------------
st.title("💺 Airfare Prediction App")

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"${mae:,.2f}")
col3.metric("RMSE", f"${rmse:,.2f}")

# -------------------------------------------------------
# Visual Diagnostics
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Actual vs Predicted", "📉 Residuals"])

with tab1:
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set_xlabel("Actual Fare")
    ax.set_ylabel("Predicted Fare")
    st.pyplot(fig)

with tab2:
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# Prediction Form – Clean & Side-by-Side
# -------------------------------------------------------
st.subheader("🔮 Predict Fare for New Route")

with st.form("prediction_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        COUPON = st.number_input("Coupon", 0.5, 3.0, 1.2)
        NEW = st.number_input("Years Since Introduced", 0, 20, 3)
        DISTANCE = st.number_input("Distance (Miles)", 100, 4000, 2000)

    with c2:
        VACATION = st.selectbox("Vacation Route?", ["No", "Yes"])
        SW = st.selectbox("Southwest Present?", ["No", "Yes"])
        PAX = st.number_input("Passengers", 1000, 50000, 12000)

    with c3:
        SLOT = st.selectbox("Slot Control", ["Free", "Controlled"])
        GATE = st.selectbox("Gate Control", ["Free", "Controlled"])
        HI = st.number_input("Hub Influence", 1000.0, 10000.0, 4000.0)

    c4, c5 = st.columns(2)

    with c4:
        S_INCOME = st.number_input("Source Income", 10000, 60000, 28000)
        S_POP = st.number_input("Source Population", 100000, 10000000, 4000000)

    with c5:
        E_INCOME = st.number_input("Destination Income", 10000, 60000, 27000)
        E_POP = st.number_input("Destination Population", 100000, 10000000, 3000000)

    submitted = st.form_submit_button("Predict Fare")

# -------------------------------------------------------
# Prediction Output
# -------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "COUPON": COUPON,
        "NEW": NEW,
        "VACATION": 1 if VACATION == "Yes" else 0,
        "SW": 1 if SW == "Yes" else 0,
        "HI": HI,
        "S_INCOME": S_INCOME,
        "E_INCOME": E_INCOME,
        "S_POP": S_POP,
        "E_POP": E_POP,
        "SLOT": 1 if SLOT == "Controlled" else 0,
        "GATE": 1 if GATE == "Controlled" else 0,
        "PAX": PAX,
        "DISTANCE": DISTANCE
    }])

    fare = model.predict(input_df)[0]
    st.success(f"💰 Predicted Airfare: **${fare:,.2f}**")

