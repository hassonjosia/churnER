import pandas as pd
import numpy as np

# import model final
from xgboost import XGBClassifier 

# load model
import pickle
import joblib

import streamlit as st

# ----------------------------------------

# Judul
st.write("""
    <div style="text-align: center;"> 
         <h2> Customer Churn Prediction </h2>
    </div>

""", unsafe_allow_html=True)

# side bar menu for input
st.sidebar.header("Input Your Data Here")

# nuntuk input data
def user_input_features():
    credit = st.sidebar.slider(label= 'Credit Score', 
                    min_value = 350,
                    max_value = 850, value = 500)

    balance = st.sidebar.slider(label = 'Balance',
                            min_value = 0,
                            max_value = 260000, 
                            value = 130000)

    salary = st.sidebar.slider(label = 'EstimatedSalary',
                            min_value = 11,
                            max_value = 200000, 
                            value = 100000)

    age = st.sidebar.number_input(label = 'Age',
                            min_value = 18,
                            max_value = 92, 
                            value = 30)

    tenure = st.sidebar.number_input(label = 'Tenure',
                            min_value = 0,
                            max_value = 10, 
                            value = 5)

    prods = st.sidebar.number_input(label = 'NumOfProducts', 
                        min_value = 1,
                        max_value = 5,
                        value = 1)

    # untuk input data categorical

    card = st.sidebar.selectbox(label = 'HasCrCard', 
                        options = [0, 1])

    member = st.sidebar.selectbox(label = 'IsActiveMember', 
                        options = [0, 1])

    gender = st.sidebar.selectbox(label = 'Gender', 
                        options = ['Female', 'Male'])

    geo = st.sidebar.selectbox(label = 'Geography', 
                        options = ['France', 'Germany', 'Spain'])
    df = pd.DataFrame()
    df["CreditScore"] = [credit]
    df['Geography'] = [geo]
    df['Gender'] = [gender]
    df['Age'] = [age]
    df['Tenure'] = [tenure]
    df['Balance'] = [balance]
    df['NumOfProducts'] = [prods]
    df['HasCrCard'] = [card]
    df['IsActiveMember'] = [member]
    df['EstimatedSalary'] = [salary]

    return df

df_feature = user_input_features()

# memanggil model
model = joblib.load('model_xgboost_joblib')
# predict
pred = model.predict(df_feature)


# untuk membuat layout menjadi 2 bagian
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Characteristics")
    st.write(df_feature.transpose())

with col2:
    st.subheader("Predicted Result")
    if pred[0] == 1:
        st.write("<h1 style ='color : red;'> CHURN</h1>", unsafe_allow_html=True)
    else:
        st.write("<h1 style ='color : green;'> STAY</h1>", unsafe_allow_html=True)


# cek ngeluarin data inputan
# st.write(df_feature.transpose())
    