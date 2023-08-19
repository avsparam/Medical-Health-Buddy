import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    diabetes_dataset = pd.read_csv('Dataset/diabetes.csv')
    X = diabetes_dataset.drop(columns='Outcome', axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    std_data = scaler.transform(input_data_reshaped)

    data_df = pd.DataFrame(data=std_data, columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                                   'SkinThickness', 'Insulin', 'BMI',
                                                   'DiabetesPedigreeFunction', 'Age'])

    classifier = pickle.load(open('Models/diabetes_model.pkl', "rb"))
    prediction = classifier.predict(data_df)

    if prediction[0] == 0:
        return 'not diabetic'
    else:
        return 'diabetic'


st.markdown('# Diabetes Prediction')
st.markdown('*The model used here is trained on a dataset of females at least 21 years old of Pima Indian heritage.*')

parameters = ['Pregnancies', 'Glucose', 'Blood Pressure',
              'Skin Thickness', 'Insulin', 'BMI',
              'Diabetes Pedigree Function', 'Age']
parameters_help = ['Number of times pregnant',
                   'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
                   'Diastolic blood pressure (mm Hg)',
                   'The thickness of the skin fold on the triceps area of the arm (mm)',
                   'The level of insulin in the blood 2 hours after a meal (mu U/ml)',
                   'Body mass index (weight in kg/(height in m)^2)',
                   'A function that quantifies the likely hereditary risk of diabetes based on family history.',
                   'Age (years)']
input_data = [5, 166, 72, 19, 175, 25.8, 0.587, 51]
for i, parameter in enumerate(parameters):
    input_data[i] = st.text_input(label=parameter, value=input_data[i], help=parameters_help[i])

if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f'The person is **{prediction}**')
