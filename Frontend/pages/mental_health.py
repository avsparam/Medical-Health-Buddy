import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
st.header('Mental Health Disorder Prediction')
df = pd.read_csv('Dataset/mental_health.csv')

# Train model
y = df['Disorder']
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

X = df.drop(['Disorder', 'Unnamed: 25'], axis=1)

model_hgbc = HistGradientBoostingClassifier()
model_hgbc.fit(X, y)

model_rc = RidgeClassifier()
model_rc.fit(X, y)

model_mlpc = MLPClassifier(alpha=1, max_iter=1000)
model_mlpc.fit(X, y)

model_dtc = DecisionTreeClassifier(max_depth=5)
model_dtc.fit(X, y)

model_rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model_rfc.fit(X, y)

# Input features  
st.subheader('Select your symptoms')
features = X.columns.tolist()
input_data = st.multiselect(label="Symptoms", options=features, max_selections=6)
print(input_data)

# Reshape
x_encoder = LabelEncoder()
labels = x_encoder.fit_transform(features)
input_enc = x_encoder.transform(input_data)
input = [0 for i in range(24)]
for i in range(0, len(input_enc)):
    input[np.where(labels == (input_enc[i]))[0][0]] = 1
input = np.array(input)
input = input.reshape(1, -1)
print(input)

# Predict 
if st.button("Predict"):
    prediction_hgbs = model_hgbc.predict(input)
    prediction_rc = model_rc.predict(input)
    prediction_mlpc = model_mlpc.predict(input)
    prediction_dtc = model_dtc.predict(input)
    prediction_rfc = model_rfc.predict(input)

    print(prediction_hgbs, prediction_rc, prediction_mlpc, prediction_dtc, prediction_rfc)

    st.subheader('Predictions using different models:')
    st.write("HistGradientBoostingClassifier: " + y_encoder.inverse_transform(prediction_hgbs)[0])
    st.write("RidgeClassifier: " + y_encoder.inverse_transform(prediction_rc)[0])
    st.write("MLPClassifier: " + y_encoder.inverse_transform(prediction_mlpc)[0])
    st.write("DecisionTreeClassifier: " + y_encoder.inverse_transform(prediction_dtc)[0])
    st.write("RandomForestClassifier: " + y_encoder.inverse_transform(prediction_rfc)[0])