#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page title
st.title("🎈 Simple Streamlit Demo App")

# Input section
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)

if st.button("Submit"):
    st.success(f"Hello {name}! You are {age} years old.")
    st.balloons()

# Example data and chart
st.subheader("📊 Random Data Visualization")

data = pd.DataFrame({
    'x': np.arange(1, 11),
    'y': np.random.randint(10, 100, 10)
})

st.write("Here’s a random dataset:")
st.dataframe(data)

fig, ax = plt.subplots()
ax.plot(data['x'], data['y'], marker='o', color='teal')
ax.set_title("Random Line Chart")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
st.pyplot(fig)


# In[ ]:




