import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

st.set_page_config(page_title='Dummy variable and one hot encoding',page_icon='D:\Machine_learning\one_hot_encoding\logo.png')

st.title("One Hot Encoding")
st.write('''Exercise:
At the same level as this notebook on github, there is an Exercise folder that contains carprices.csv. This file has car sell prices for 3 different models. First plot data points on a scatter plot chart to see if linear regression model can be applied. If yes, then build a model that can answer following questions,

1) Predict price of a mercedez benz that is 4 yr old with mileage 45000

2) Predict price of a BMW X5 that is 7 yr old with mileage 86000

3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())''')

st.image("D:\Machine_learning\one_hot_encoding\Default_create_a_best_image_for_my_desktop_wallpaper_use_any_i_0.jpg")

df = pd.read_csv("D:\Machine_learning\one_hot_encoding\carprices.csv")
st.subheader("Data Set")
st.table(df)


st.write("Perfoming one hot encoding and updating dataframe")
st.code('''
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(drop = 'first', handle_unknown='error',sparse_output=False)
data = ohe.transform(df[['Car Model']])
data = pd.DataFrame(data)
pd.concat([df,data],axis='columns')''')

ohe = joblib.load("D:\\Machine_learning\\one_hot_encoding\\onhot_encoder.joblib")
data = ohe.transform(df[['Car Model']])
data = pd.DataFrame(data,columns=['col1','col2'])

df = df.drop('Car Model',axis='columns')
df = pd.concat([data,df],axis='columns')
st.table(df)

reg = joblib.load("D:\\Machine_learning\\one_hot_encoding\\reg.joblib")

def format(str,mil,age):
    str = pd.DataFrame(ohe.transform([[str]]),columns=['col1','col2'])
    print(str)
    data = pd.DataFrame([[mil,age]],columns=['Mileage','Age(yrs)'])
    return pd.concat([str,data],axis='columns')

st.subheader("Solutions:")
st.write("1) Predict price of a mercedez benz that is 4 yr old with mileage 45000")
st.write(reg.predict(format('Mercedez Benz C class',45000,4)))

st.write("2) Predict price of a BMW X5 that is 7 yr old with mileage 86000")
st.write(reg.predict(format('BMW X5',86000,7)))

st.write("3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())")
st.write(reg.score(df[['col1','col2','Mileage','Age(yrs)']],df[['Sell Price($)']]),'in-sample evaluation')

st.subheader("Predictions")
op = ['BMW X5','Audi A5','Mercedez Benz C class']
car = st.radio(label="Car Model",options=op)
mil = st.number_input(label='Mileage',step=1)
age = st.number_input(label='Age(yrs)',step=1)

btn = st.button(label="Predict")

if btn:
    if car == '' or car not in op:
        st.warning('Enter a valid car model')

    else:    
        st.write(reg.predict(format(car,mil,age)))