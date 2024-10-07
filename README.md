# Car Price Prediction using One-Hot Encoding and Linear Regression

This project uses one-hot encoding and linear regression to predict car prices based on their mileage and age. The analysis is visualized using a Streamlit app.

## Requirements

- streamlit
- pandas
- numpy
- matplotlib
- joblib

## Dataset

The dataset used is `carprices.csv`, which contains information about car models, their mileage, age, and sell prices.

## Steps

1. **Data Preparation**: Load the dataset and preprocess it by performing one-hot encoding on the categorical features.
2. **Feature Analysis**: Analyze the features that influence car prices using scatter plots.
3. **Model Training**: Train a linear regression model to predict car prices.
4. **Visualization**: Visualize the data and model predictions using Streamlit.

## Streamlit App

The Streamlit app provides an interactive interface for exploring the data and model predictions.

### Installation

To run the Streamlit app, you'll need to install the required packages:

```sh
pip install streamlit pandas numpy matplotlib joblib
