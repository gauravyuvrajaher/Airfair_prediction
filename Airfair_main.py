#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns  # Optional, can remove if not needed

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
    vacation_map = {"Yes": 1, "No": 0}
    slot_map = {"Controlled": 1, "Free": 0}
    gate_map = {"Controlled": 1, "Free": 0}

    route_data = {
        "const": 1,
        "COUPON": COUPON,
        "NEW": NEW,
        "VACATION": vacation_map[VACATION],
        "SW": 0,
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

    fare_no_sw = model.predict(X_new)[0]

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
    # 5. Visualization Tabs using Streamlit charts
    # ----------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Actual vs Predicted", "📉 Residuals", "📦 Residual Distribution"])

    with tab1:
        st.subheader("Actual vs Predicted Fare")
        chart_data = df[["FARE", "Predicted_FARE"]]
        st.line_chart(chart_data)  # Simple line chart

    with tab2:
        st.subheader("Residuals vs Predicted Fare")
        chart_data = df[["Predicted_FARE", "Residuals"]].rename(columns={"Predicted_FARE":"x","Residuals":"y"})
        st.bar_chart(chart_data.set_index("x"))

    with tab3:
        st.subheader("Distribution of Residuals")
        hist_data = pd.DataFrame(df["Residuals"])
        st.bar_chart(hist_data)  # Simple bar chart for residual distribution

else:
    st.info("👈 Enter values in the sidebar and click **Predict Fare** to see results.")

# ----------------------------
# 6. Model Summary (Optional)
# ----------------------------
with st.expander("📜 Model Summary"):
    st.text(model.summary())


# In[ ]:




