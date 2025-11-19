#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airfare Prediction App", layout="wide")

# ----------------------------
# 1. Load and prepare data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Airfares_data.xlsx")

    # Encode categorical variables
    df["VACATION"] = df["VACATION"].map({"Yes": 1, "No": 0})
    df["SW"] = df["SW"].map({"Yes": 1, "No": 0})
    df["SLOT"] = df["SLOT"].map({"Controlled": 1, "Free": 0})
    df["GATE"] = df["GATE"].map({"Controlled": 1, "Free": 0})

    # Define X and y
    X = df[["COUPON", "NEW", "VACATION", "SW", "HI", "S_INCOME",
             "E_INCOME", "S_POP", "E_POP", "SLOT", "GATE", "PAX", "DISTANCE"]]
    y = df["FARE"]

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Add predictions and residuals
    df["Predicted_FARE"] = model.predict(sm.add_constant(X))
    df["Residuals"] = df["FARE"] - df["Predicted_FARE"]

    return df, model


df, model = load_data()

# ----------------------------
# 2. Sidebar Inputs
# ----------------------------
st.sidebar.header("✈️ Enter Route Details")

COUPON = st.sidebar.number_input("COUPON", value=1.202, step=0.001)
NEW = st.sidebar.number_input("NEW (Years since route introduced)", value=3)
VACATION = st.sidebar.selectbox("VACATION", ["No", "Yes"])
HI = st.sidebar.number_input("HI (Hub Influence Index)", value=4442.141, step=0.01)
S_INCOME = st.sidebar.number_input("Source Income", value=28760)
E_INCOME = st.sidebar.number_input("Destination Income", value=27664)
S_POP = st.sidebar.number_input("Source Population", value=4557004)
E_POP = st.sidebar.number_input("Destination Population", value=3195503)
SLOT = st.sidebar.selectbox("Slot Controlled?", ["Free", "Controlled"])
GATE = st.sidebar.selectbox("Gate Controlled?", ["Free", "Controlled"])
PAX = st.sidebar.number_input("Passengers", value=12782)
DISTANCE = st.sidebar.number_input("Distance (miles)", value=1976)

# ----------------------------
# 3. Main Title
# ----------------------------
st.title("💺 Airfare Prediction App")
st.write("Predict average airfare for a new route based on route and city characteristics.")

# ----------------------------
# 4. Predict Fare on Button Click
# ----------------------------
if st.button("🔮 Predict Fare"):
    # Encode categorical inputs
    vacation_map = {"Yes": 1, "No": 0}
    slot_map = {"Controlled": 1, "Free": 0}
    gate_map = {"Controlled": 1, "Free": 0}

    # Prepare input data
    route_data = {
        "const": 1,
        "COUPON": COUPON,
        "NEW": NEW,
        "VACATION": vacation_map[VACATION],
        "SW": 0,  # Start without Southwest
        "HI": HI,
        "S_INCOME": S_INCOME,
        "E_INCOME": E_INCOME,
        "S_POP": S_POP,
        "E_POP": E_POP,
        "SLOT": slot_map[SLOT],
        "GATE": gate_map[GATE],
        "PAX": PAX,
        "DISTANCE": DISTANCE
    }

    X_new = pd.DataFrame([route_data])

    # Predict without Southwest
    fare_no_sw = model.predict(X_new)[0]

    # Predict with Southwest
    X_new_sw = X_new.copy()
    X_new_sw["SW"] = 1
    fare_with_sw = model.predict(X_new_sw)[0]

    fare_reduction = fare_no_sw - fare_with_sw

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Fare (No Southwest)", f"${fare_no_sw:,.2f}")
    col2.metric("Predicted Fare (With Southwest)", f"${fare_with_sw:,.2f}")
    col3.metric("Fare Reduction", f"${fare_reduction:,.2f}")

    st.markdown("---")

    # ----------------------------
    # 5. Visualization Tabs
    # ----------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Actual vs Predicted", "📉 Residuals", "📦 Residual Distribution"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=df["FARE"], y=df["Predicted_FARE"], alpha=0.7, color="royalblue", ax=ax)
        ax.plot([df["FARE"].min(), df["FARE"].max()],
                [df["FARE"].min(), df["FARE"].max()],
                'r--', linewidth=2)
        ax.set_title("Actual vs Predicted Fare")
        ax.set_xlabel("Actual Fare")
        ax.set_ylabel("Predicted Fare")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=df["Predicted_FARE"], y=df["Residuals"], color="darkorange", ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title("Residuals vs Predicted Fare")
        ax.set_xlabel("Predicted Fare")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.histplot(df["Residuals"], bins=30, kde=True, color='green', ax=ax)
        ax.set_title("Distribution of Residuals")
        ax.set_xlabel("Residuals")
        st.pyplot(fig)

else:
    st.info("👈 Enter values in the sidebar and click **Predict Fare** to see results.")

# ----------------------------
# 6. Model Summary (Optional)
# ----------------------------
with st.expander("📜 Model Summary"):
    st.text(model.summary())


# In[ ]:




