
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt

import kmeans_Clustering as Kmeans

# headings
month = datetime.now().month
title = "options-2-trees"
if 1 <= month <= 5:
    st.title(title + " 🌳🌳")
elif 5 < month <= 8:
    st.title(title + " 🌴🌴")
elif 8 < month <= 11:
    st.title(title + " 🌲🌲")
else:
    st.title(title + " 🎄🎄")
st.write("by [Tony](https://www.linkedin.com/in/tony-c-8b592b162/)")
st.sidebar.title("Parameters")

# user inputs on sidebar
S = st.sidebar.slider('Stock Price (S)', value=500, 
                      min_value=250, max_value=750)

X = st.sidebar.slider('Exercise Price (X)', value=500, 
                      min_value=250, max_value=750)

T = st.sidebar.slider('Time Periods (T)', value=5, 
                      min_value=1, max_value=15)

r = st.sidebar.slider('Inter-period Interest Rate (r)', value=0.0, 
                      min_value=0.0, max_value=0.05, step=0.01)

u = st.sidebar.slider('Stock Growth Factor (u)', value=1.10, 
                      min_value=1.01, max_value=1.25, step=0.01)
d = 1/u
st.sidebar.write("Stock Decay Factor (d) ", round(d, 4))

# back to main body
st.header("*See how the Cox-Ross-Rubinstein (CRR) options pricing model react to changing parameters*")
st.markdown("This visualisation aims to explore the dynamics of abstract financial theory. "
            "Can you adjust the display to see how how the value of a call option today is positively correlated to the interest rate, "
            "or how the diffusion of the CRR tree is inherently lognormal?"
            )

st.subheader('Key:')
st.markdown("✅ Stock tree: black")
call = st.checkbox('Call tree: blue')
put = st.checkbox('Put tree: red')

# plot stock tree

# text section
st.header("1. What's Going On?")
st.markdown("Options are financial instruments that derive value from an underlying asset. "
            "Most often, the underlying asset is a stock, which is what we're modelling here. "
            "Options must be exercised before they expire, but can only be exercised past a certain the exercise price. "
            "There are different kinds of options. Here, we're modelling vanilla European options, which can only be exercised at expiry. "
            )

st.subheader("1.1 Main takeaways: ")
st.markdown("1) The nodes on the black tree tell us the future prices of a stock, and the nodes on the blue and red trees tell us the values of the corresponding options")

st.subheader("2.2 Formulae Used")
st.latex(r"d = \frac{1}{u}")
st.latex(r"S_{t+1} = S_t \cdot u")
st.latex(r"S_{t+1} = S_t \cdot d")
st.latex("C_T = max(S_T-X, 0)")
st.latex("P_T = max(X-S_T, 0)")
st.latex(r"p = \frac{e^r - d}{u - d}")
st.latex(r"C_t = e^{-r} (p \cdot Cu_{t+1} + (1 - p) \cdot Cd_{t+1})")
st.latex(r"P_t = e^{-r} (p \cdot Pu_{t+1} + (1 - p) \cdot Pd_{t+1})")
st.write("\n")

st.write("All information on options-2-trees is provided for educational purposes only and does not constitute "
         "financial advice."
        )
