import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def predict(input_data):
    inp = np.asarray(input_data).reshape(1, -1)

    heart_data = pd.read_csv('Dataset/heart_disease_data.csv')

    X = heart_data.drop(columns='target', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)

    inp = scaler.transform(inp)

    print(inp)

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

parameters = {'Age': 1, 'Sex': 2, 'Type of Chest Pain': 3, 'Resting Blood Pressure (in mm Hg)': 4,
              'Serum Cholesterol (in mg/dl)': 5, 'Fasting Blood Sugar (>120 mg/dl)': 6,
              'Resting Electrocardiograph': 7, 'Maximum Heart Rate': 8, 'Exercise Induced Angina': 9,
              'ST depression induced by exercise relative to rest': 10,
              'The slope of the peak exercise ST segment': 11, 'Number of major vessels colored by fluoroscopy': 12,
              'Thallium-201 (Myocardial Perfusion Imaging)': 13
              }

numerical_parameters = ['Age', 'Resting Blood Pressure (in mm Hg)',
                        'Serum Cholesterol (in mg/dl)', 'Maximum Heart Rate',
                        'ST depression induced by exercise relative to rest',
                        'Number of major vessels colored by fluoroscopy']

numerical_parameters_help = ['**Your age in years.**',
                             '**...** Normal Range: 5 to 20 mg/dl',
                             '**...** Normal Range: 0.74 to 1.35 mg/dL [men]; 0.59 to 1.04 mg/dL [women]',
                             '**...** Normal Range: 4% to 5.6%',
                             '**...** Normal Range: 125 to 200mg/dL',
                             '**...** Normal Range: Less than 150 mg/dL'
                             ]

categorical_parameters = ['Sex', 'Type of Chest Pain', 'Fasting Blood Sugar (>120 mg/dl)',
                          'Resting Electrocardiograph', 'Exercise Induced Angina',
                          'The slope of the peak exercise ST segment', 'Thallium-201 (Myocardial Perfusion Imaging)']

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

categorical_parameters_help = ['**Your gender.**',
                               '**...** Normal Range: 5 to 20 mg/dl',
                               '**...** Normal Range: 0.74 to 1.35 mg/dL [men]; 0.59 to 1.04 mg/dL [women]',
                               '**...** Normal Range: 4% to 5.6%',
                               '**...** Normal Range: 125 to 200mg/dL',
                               '**...** Normal Range: Less than 150 mg/dL',
                               '**...** Normal Range: Less than 150 mg/dL'
                               ]

input_data = [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]

for i, parameter in enumerate(categorical_parameters):
    input_data[parameters[parameter] - 1] = categorical_parameters_options[i][
        st.selectbox(label=parameter,
                     options=categorical_parameters_options[i].keys(),
                     index=input_data[parameters[parameter] - 1],
                     help=categorical_parameters_help[i])]

for i, parameter in enumerate(numerical_parameters):
    input_data[parameters[parameter] - 1] = st.text_input(label=parameter,
                                                          value=input_data[parameters[parameter] - 1],
                                                          help=numerical_parameters_help[i])

if st.button("Predict"):
    numerical_input_data = []
    for value in input_data:
        numerical_input_data.append(float(value))
    prediction = predict(numerical_input_data)
    st.markdown(f'# The person {prediction}')
