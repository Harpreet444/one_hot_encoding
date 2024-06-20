import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib

st.set_page_config(page_title='Dummy variable and one hot encoding',page_icon='logo.png',layout='wide')

co1, co2, co3 = st.columns([1,3,1])

co2.title("One Hot Encoding")
co2.write('''Exercise:
At the same level as this notebook on github, there is an Exercise folder that contains carprices.csv. This file has car sell prices for 3 different models. First plot data points on a scatter plot chart to see if linear regression model can be applied. If yes, then build a model that can answer following questions,

1) Predict price of a mercedez benz that is 4 yr old with mileage 45000

2) Predict price of a BMW X5 that is 7 yr old with mileage 86000

3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())''')


co2.image("Default_create_a_best_image_for_my_desktop_wallpaper_use_any_i_0.jpg")



df = pd.read_csv("carprices.csv",)

co2.subheader("Dataset")
co2.table(df)

co2.subheader("Scatter Plot")
fig, ax = plt.subplots()
ax.scatter(df['Mileage'], df['Sell Price($)'], color='g', label="Scatter plot",marker = '+')  # Experience vs. Salary
ax.set_xlabel('Mileage')
ax.set_ylabel('Sell Price($)')
ax.set_xlabel('Mileage')

co2.pyplot(fig)


co2.write("Perfoming one hot encoding and updating dataframe")
co2.code('''
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(drop = 'first', handle_unknown='error',sparse_output=False)
data = ohe.transform(df[['Car Model']])
data = pd.DataFrame(data)
pd.concat([df,data],axis='columns')''')

ohe = joblib.load("onhot_encoder.joblib")
data = ohe.transform(df[['Car Model']])
data = pd.DataFrame(data,columns=['col1','col2'])

df = df.drop('Car Model',axis='columns')
df = pd.concat([data,df],axis='columns')
co2.table(df)

reg = joblib.load("reg.joblib")

def format(str,mil,age):
    str = pd.DataFrame(ohe.transform([[str]]),columns=['col1','col2'])
    print(str)
    data = pd.DataFrame([[mil,age]],columns=['Mileage','Age(yrs)'])
    return pd.concat([str,data],axis='columns')

co2.subheader("Solutions:")
co2.write("1) Predict price of a mercedez benz that is 4 yr old with mileage 45000")
co2.write(reg.predict(format('Mercedez Benz C class',45000,4)))

co2.write("2) Predict price of a BMW X5 that is 7 yr old with mileage 86000")
co2.write(reg.predict(format('BMW X5',86000,7)))

co2.write("3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())")
co2.write(reg.score(df[['col1','col2','Mileage','Age(yrs)']],df[['Sell Price($)']]))

co2.subheader("Predictions")
op = ['BMW X5','Audi A5','Mercedez Benz C class']
car = co2.radio(label="Car Model",options=op)
mil = co2.number_input(label='Mileage',step=1)
age = co2.number_input(label='Age(yrs)',step=1)

btn = co2.button(label="Predict")

if btn:
    if car == '' or car not in op:
        co2.warning('Enter a valid car model')

    else:    
        co2.write(reg.predict(format(car,mil,age)))
