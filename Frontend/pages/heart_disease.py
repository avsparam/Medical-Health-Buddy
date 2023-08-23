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

parameters = {'Age': 1, 'Sex': 2, 'Type of Chest Pain': 3, 'Resting Systolic Blood Pressure (in mm Hg)': 4,
              'Serum Cholesterol (in mg/dl)': 5, 'Fasting Blood Sugar (>120 mg/dl)': 6,
              'Resting Electrocardiograph': 7, 'Maximum Heart Rate': 8, 'Exercise Induced Angina': 9,
              'ST depression induced by exercise relative to rest': 10,
              'The slope of the peak exercise ST segment': 11, 'Number of major vessels colored by fluoroscopy': 12,
              'Thallium-201 (Myocardial Perfusion Imaging)': 13
              }

numerical_parameters = ['Age', 'Resting Systolic Blood Pressure (in mm Hg)',
                        'Serum Cholesterol (in mg/dl)', 'Maximum Heart Rate',
                        'ST depression induced by exercise relative to rest'
                        ]

numerical_parameters_help = ['**Your age in years.**',
                             '''The top number in the blood pressure reading and represents the pressure in the 
                             arteries when the heart contracts or beats, pumping blood into the circulation. It is 
                             the maximum pressure exerted on the arterial walls during each heartbeat.  
                             Normal Range: <120 mm Hg''',
                             '''Serum cholesterol refers to the level of cholesterol present in the blood.  
                             Normal Range: Less than 200 mg/dL''',
                             '''Maximum heart rate (MHR) refers to the highest number of beats per minute (BPM) that 
                             your heart can reach during physical exertion.  
                             Normal Range: 220 - age''',
                             '''ST depression induced by exercise relative to rest refers to the change in the ST 
                             segment of an electrocardiogram (ECG) during physical activity compared to the resting 
                             state.'''
                             ]

categorical_parameters = ['Sex', 'Type of Chest Pain', 'Fasting Blood Sugar (>120 mg/dl)',
                          'Resting Electrocardiograph', 'Exercise Induced Angina',
                          'The slope of the peak exercise ST segment', 'Number of major vessels colored by fluoroscopy',
                          'Thallium-201 (Myocardial Perfusion Imaging)'
                          ]

categorical_parameters_options = [{'Female': 0, 'Male': 1},
                                  {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4},
                                  {'False': 0, 'True': 1},
                                  {'Normal': 0,
                                   'ST-T wave have abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,
                                   'Showing probable or definite left ventricular hypertrophy by Estes criteria': 2},
                                  {'No': 0, 'Yes': 1},
                                  {'Up Slope': 1, 'Flat': 2, 'Down Slope': 3},
                                  {'None': 0, 'One': 1, 'Two': 2, 'Three': 3},
                                  {'Thal 3: Normal': 3, 'Thal 6: Fixed defect': 6, 'Thal 7: Reversable defect': 7}
                                  ]

categorical_parameters_help = ['Your **gender**.',
                               '''**Typical Angina:** This type of chest pain is also known as "stable angina" and is 
                               characterized by a predictable pattern. The pain is usually described as a discomfort, 
                               pressure, squeezing, or heaviness in the chest.  
                               **Atypical Angina:** Atypical angina refers to chest pain that does not fit the 
                               typical pattern of stable angina. The pain may be less predictable and may not be 
                               relieved by rest or nitroglycerin. It can manifest as sharp, stabbing, burning, 
                               or fleeting discomfort in the chest.  
                               **Non-Anginal Pain:** Non-anginal chest pain refers to discomfort or pain in the chest 
                               that is unrelated to the heart or coronary artery disease. It can be caused by various 
                               factors, such as muscle strain, acid reflux, anxiety, respiratory conditions, 
                               or inflammation of the chest wall.  
                               **Asymptomatic:** Asymptomatic means the absence of any chest pain or symptoms related 
                               to the heart. Some individuals, particularly in the early stages of certain heart 
                               conditions, may not experience any chest pain or discomfort.''',
                               '''The fasting blood sugar level refers to the amount of glucose (sugar) present in 
                               the bloodstream after a period of fasting, typically for 8 to 12 hours. A fasting 
                               blood sugar level of greater than 120 mg/dL may indicate hyperglycemia, which means 
                               high blood sugar.''',
                               '''A resting electrocardiogram (ECG) is a test that measures the electrical activity 
                               of the heart while the person is at rest.  
                               A **normal resting ECG** shows a regular heart 
                               rhythm, normal heart rate, and no significant abnormalities in the electrical patterns 
                               of the heart. The ST-T wave on an ECG represents the repolarization phase of the 
                               heart's electrical cycle.  
                               **Abnormalities in the ST-T wave** can indicate different 
                               cardiac conditions. T-wave inversions, ST elevation, or ST depression of more than 
                               0.05 mV can suggest ischemia (lack of blood flow) to the heart muscle, myocardial 
                               infarction (heart attack), or other cardiac abnormalities.  
                               **Left ventricular 
                               hypertrophy** refers to the thickening of the heart's left ventricle, which is often 
                               associated with an underlying heart condition, such as high blood pressure or heart 
                               valve problems. The Estes criteria is a set of ECG criteria used to assess the 
                               likelihood of left ventricular hypertrophy. If the ECG shows specific patterns, 
                               such as increased QRS voltage or repolarization abnormalities, it can indicate 
                               probable or definite LVH.''',
                               '''**Exercise-induced angina**, also known as exertional angina, refers to chest pain or 
                               discomfort that occurs during physical activity or exercise. It is typically caused by 
                               reduced blood flow to the heart muscle due to narrowed or blocked coronary arteries.''',
                               '''The slope of the peak exercise ST segment refers to the direction or shape of the 
                               ST segment on an electrocardiogram (ECG) during exercise testing.  
                               An **up-sloping ST 
                               segment** during exercise testing refers to a gradual upward movement of the ST segment 
                               from the baseline. This can indicate a **normal response** to exercise, particularly in 
                               individuals without significant coronary artery disease.  
                               A **flat ST segment** during 
                               exercise testing refers to a horizontal or nearly horizontal ST segment that remains 
                               relatively unchanged from the baseline. This can indicate an **equivocal or 
                               indeterminate response**.  
                               A **down-sloping ST segment** during exercise testing refers to a 
                               downward or descending ST segment from the baseline. This can indicate **myocardial 
                               ischemia**, which means there is a reduced blood flow to the heart muscle during 
                               exercise.''',
                               '''The number of major vessels colored by fluoroscopy refers to the visualization of 
                             coronary arteries during a cardiac catheterization procedure. The number of major 
                             vessels colored by fluoroscopy depends on the specific coronary anatomy of the 
                             individual.''',
                               '''**Thallium-201 MPI (Myocardial Perfusion Imaging)** is commonly used in cardiology.  
                               A **Thal 3** interpretation indicates a **normal result** from the Thallium MPI test. 
                               This means that the perfusion of blood to the heart muscle appears to be normal, 
                               without any significant blockages or abnormalities.  
                               A **Thal 6** interpretation 
                               suggests a **fixed** defect in the myocardial perfusion. This means that a portion of 
                               the heart muscle is not receiving adequate blood supply, potentially due to scarring 
                               or damage caused by a previous heart attack or other cardiac conditions.  
                               A **Thal 7** 
                               interpretation indicates a **reversible defect** in the myocardial perfusion. This 
                               means that there is reduced blood flow to a portion of the heart muscle during stress 
                               or exercise, but the blood flow improves or normalizes when the stress is relieved or 
                               with the administration of certain medications.'''
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
