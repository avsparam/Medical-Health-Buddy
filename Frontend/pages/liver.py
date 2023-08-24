import numpy as np
from sklearn.preprocessing import RobustScaler
import streamlit as st
import pickle


def predict(input_data):
    inp = np.asarray(input_data).reshape(1, -1)

    scaler = RobustScaler()
    inp = scaler.fit_transform(inp)

    model = pickle.load(file=open('Models/liver_model.pkl', 'rb'))
    prediction = model.predict(inp)[0]
    return prediction


st.markdown('# Liver Disease Prediction')
st.markdown('''*The model used here is trained on a dataset containing laboratory analysis of about 600 patients from 
            north east of Andhra Pradesh, India.*  
            http://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset''')

input_data = [65, 1, 0.7, 187, 16, 0.9]

parameters = {'Age': 0, 'Gender': 1, 'Total Bilirubin': 2, 'Alkaline Phosphotase': 3, 'Alamine Aminotransferase': 4,
              'Albumin and Globulin Ratio': 5}
numerical_parameters = ['Age', 'Total Bilirubin', 'Alkaline Phosphotase',
                        'Alamine Aminotransferase', 'Albumin and Globulin Ratio'
                        ]

numerical_parameters_help = ['**Your age in years.**',
                             '''Total bilirubin is a blood test that measures the level of bilirubin in the 
                             bloodstream. Bilirubin is a yellowish pigment produced by the breakdown of red blood 
                             cells in the body.  
                             Normal Range: 0.3 to 1.2 mg/dL''',
                             '''Alkaline phosphatase (ALP) is an enzyme found in various tissues throughout the body, 
                             including the liver, bones, intestines, and kidneys. Elevated levels of alkaline 
                             phosphatase can indicate various conditions like Liver diseases, Bone disorders, 
                             Biliary obstruction and Intestinal disorders.  
                             Normal Range: 20 to 140 international units per liter (IU/L)''',
                             '''Alanine aminotransferase (ALT), also known as serum glutamic-pyruvic transaminase (
                             SGPT), is an enzyme found predominantly in the liver. ALT is commonly measured through a 
                             blood test and is used as a marker to assess liver health and function.  
                             Normal Range: 7 and 55 units per liter (U/L) of blood''',
                             '''The albumin and globulin ratio (A/G ratio) is a calculated value that compares the 
                             levels of albumin and globulin in the blood. Albumin and globulin are two types of 
                             proteins found in the blood, and their ratio can provide information about various 
                             health conditions.  
                             Normal Range: greater than 1'''
                             ]

categorical_parameters = ['Gender',]

categorical_parameters_options = [{'Male': 0, 'Female': 1},]

categorical_parameters_help = ['Your **gender**.',]

for i, parameter in enumerate(categorical_parameters):
    input_data[parameters[parameter]] = categorical_parameters_options[i][
        st.selectbox(label=parameter,
                     options=categorical_parameters_options[i].keys(),
                     index=input_data[parameters[parameter]],
                     help=categorical_parameters_help[i])]

for i, parameter in enumerate(numerical_parameters):
    input_data[parameters[parameter]] = st.text_input(label=parameter,
                                                          value=input_data[parameters[parameter]],
                                                          help=numerical_parameters_help[i])

if st.button("Predict"):
    numerical_input_data = []
    for value in input_data:
        numerical_input_data.append(float(value))

    prediction = predict(numerical_input_data)
    if prediction == 1:
        st.markdown('Patient is suffering from liver disease.')
    elif prediction == 2:
        st.markdown('Patient is not suffering from liver disease.')
