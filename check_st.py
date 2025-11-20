#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st

# Title and subtitle
st.title("💡 My First Streamlit App")
st.subheader("No external libraries — just Streamlit!")

# Input section
name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)

# Button to trigger output
if st.button("Say Hello"):
    st.success(f"Hello, {name}! You are {age} years old. 🎉")
else:
    st.info("👈 Enter your name and age, then click 'Say Hello'.")

# A simple checkbox and slider
agree = st.checkbox("I like Streamlit!")
level = st.slider("How much do you like it?", 0, 100, 50)

if agree:
    st.write(f"You like Streamlit {level}% 💪")
else:
    st.write("Try checking the box above 👆")


# In[ ]:




