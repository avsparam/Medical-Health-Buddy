import pandas as pd
import streamlit as st
import numpy as np
import pickle


def predict(input_data):
    inp = np.asarray(input_data).reshape(1, -1)

    model = pickle.load(open('Models/heart_disease_model.pkl', 'rb'))
    prediction = model.predict(inp)[0]

    if prediction == 0:
        return 'does not have a Heart Disease'
    else:
        return 'has Heart Disease'


st.markdown('# Heart Disease Prediction')
st.markdown('*The model used here is trained on a dataset containing laboratory analysis of about 300 patients. '
            'Patients were classified as having or not having heart disease based on cardiac catheterization, '
            'the gold standard. If they had more than 50% narrowing of a coronary artery they were labeled as having '
            'heart disease.* '
            'https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')

numerical_parameters = ['Age', 'Resting Blood Pressure (in mm Hg)',
                        'Serum Cholesterol (in mg/dl)', 'Maximum Heart Rate',
                        'ST depression induced by exercise relative to rest',
                        'Number of major vessels colored by fluoroscopy',]

categorical_parameters = ['Sex', 'Type of Chest Pain', 'Fasting Blood Sugar (>120 mg/dl)',
                          'Resting Electrocardiograph', 'Exercise Induced Angina',
                          'The slope of the peak exercise ST segment', 'thal']

categorical_parameters_options = [{'female': 0, 'male': 1},
                                  {'typical angina': 1, 'atypical angina': 2, 'non-anginal pain': 3, 'asymptomatic': 4},
                                  {'false': 0, 'true': 1},
                                  {'normal': 0,
                                   'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,
                                   'showing probable or definite left ventricular hypertrophy by Estes criteria': 2},
                                  {'no': 0, 'yes': 1},
                                  {'upsloping': 1, 'flat': 2, 'downsloping': 3},
                                  {'normal': 3, 'fixed defect': 6, 'reversable defect': 7}
                                  ]

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

input_data = [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]

# input_data[0] = {'Male': 0, 'Female': 1}[
#     st.selectbox(label='Gender', options=['Male', 'Female'], index=1, help='Enter your gender')]
#
# for i, parameter in enumerate(parameters):
#     input_data[i + 1] = st.text_input(label=parameter, value=input_data[i + 1], help=parameters_help[i])
#
if st.button("Predict"):
    prediction = predict(input_data)
    st.markdown(f'# The person {prediction}')
