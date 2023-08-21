import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


def predict(input_data):
    inp = np.asarray(input_data).reshape(1, -1)

    diabetes_dataset = pd.read_csv('Dataset/diabetes_new.csv')
    X = diabetes_dataset.drop(columns=['CLASS', 'No_Pation', 'ID'], axis=1)

    gender_dict = {'M': 0, 'F': 1}
    for i, ele in enumerate(X['Gender']):
        X['Gender'][i] = gender_dict[ele.strip().upper()]

    scaler = StandardScaler()
    scaler.fit(X)
    std_data = scaler.transform(inp)

    classifier = pickle.load(open('Models/diabetes_model_new.pkl', "rb"))
    prediction = classifier.predict(std_data)[0]

    if prediction == 0:
        return 'diabetic'
    elif prediction == 1:
        return 'pre-diabetic'
    elif prediction == 2:
        return 'not diabetic'


st.markdown('# Diabetes Prediction')
st.markdown('*The model used here is trained on a dataset containing medical information and laboratory analysis of '
            'Iraqi patients. This data was acquired from the laboratory of Medical City Hospital and the Specialized '
            'Center for Endocrinology and Diabetes (Al-Kindi Teaching Hospital).* '
            'https://data.mendeley.com/datasets/wj9rwkp9c2/1')

parameters = ['Age', 'Urea', 'Creatinine ratio', 'HBA1C',
              'Cholesterol', 'Triglycerides', 'HDL', 'LDL',
              'VLDL', 'Body Mass Index']

parameters_help = ['**Your age in years.**',
                   '**Level of urea in your blood.** Normal Range: 5 to 20 mg/dl',
                   '**The ratio of creatinine in your urine to the creatinine level in your blood.** Normal Range: 0.74 to 1.35 mg/dL [men]; 0.59 to 1.04 mg/dL [women]',
                   '**Your average blood glucose (sugar) levels for the last two to three months.** Normal Range: 4% to 5.6%',
                   '**Your total cholesterol level, which is a fatty substance found in your blood.** Normal Range: 125 to 200mg/dL',
                   '**Your triglyceride level, which is a type of fat found in your blood.** Normal Range: Less than 150 mg/dL',
                   '**High-Density Lipoprotein, often referred to as "good" cholesterol.** Normal Range: 35 to 65 mg/dL [men], 35 to 80 mg/dL [women]',
                   '**Low-Density Lipoprotein, often referred to as "bad" cholesterol.** Normal Range: Less than 100mg/dL',
                   '**Very Low-Density Lipoprotein, which is a type of lipoprotein that carries triglycerides in the blood.** Normal Range: 2 to 30 mg/dL',
                   '**Body mass index (weight in kg/(height in m)^2).** Normal Range: 18.5 to 24.9']

# input_data = [1, 50, 4.7, 46, 4.9, 4.2, 0.9, 2.4, 1.4, 0.5, 24.0]
input_data = [0, 34, 3.9, 81, 6, 6.2, 3.9, 0.8, 1.9, 1.8, 23]

input_data[0] = {'Male': 0, 'Female': 1}[
    st.selectbox(label='Gender', options=['Male', 'Female'], index=1, help='Enter your gender')]

for i, parameter in enumerate(parameters):
    input_data[i + 1] = st.text_input(label=parameter, value=input_data[i + 1], help=parameters_help[i])

if st.button("Predict"):
    prediction = predict(input_data)
    st.markdown(f'# The person is {prediction}')
