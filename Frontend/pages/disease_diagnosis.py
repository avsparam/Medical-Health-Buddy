import streamlit as st
import pandas as pd
import pickle
from tensorflow import keras
import numpy as np


MAX_SYMPTOMS = 4


def show_prediction(input_values):
    predictions=create_input_df(input_values)
    st.write(f'''# Predicted Disease using different models\n
    LR: {predictions[0]}\n
    KNN: {predictions[1]}\n
    DT: {predictions[2]}\n
    DL: {predictions[3]}''')


def create_input_df(input_symptoms):
    predictions = []
    input_symptoms_df = pd.DataFrame([input_symptoms_dict.values()], columns=symptoms_list)
    for symptom in input_symptoms:
        input_symptoms_df[symptom] = 1

    prediction_lr = model_lr.predict(input_symptoms_df)[0]
    predictions.append(prediction_lr)

    prediction_knn = model_knn.predict(input_symptoms_df)[0]
    predictions.append(prediction_knn)

    prediction_dt = model_dt.predict(input_symptoms_df)[0]
    predictions.append(prediction_dt)

    input_symptoms_array = np.array(input_symptoms_df, dtype=np.float32)
    prediction_dl = model_dl.predict(input_symptoms_array)[0]
    predictions.append(idx_to_disease[int(np.argmax(prediction_dl))]+' ('+str(round(prediction_dl[np.argmax(prediction_dl)]*100, 2))+'%)')
    return predictions


def get_model():
    pickle_in = open("Models/model_lr.pkl","rb")
    model_lr = pickle.load(pickle_in)

    pickle_in = open("Models/model_knn.pkl","rb")
    model_knn = pickle.load(pickle_in)

    pickle_in = open("Models/model_dt.pkl","rb")
    model_dt = pickle.load(pickle_in)

    model_dl = keras.models.load_model("Models/model_dl.keras")

    return model_lr, model_knn, model_dt, model_dl


file = open('Dataset/symptoms.txt', 'r')
symptoms = file.readlines()
file.close()

input_symptoms_dict = {}
symptom_to_idx = {}

for i, element in enumerate(symptoms):
    input_symptoms_dict[element.strip()] = 0
    symptom_to_idx[element.strip()] = i

symptoms_list = list(input_symptoms_dict.keys())

file = open('Dataset/diseases.txt', encoding='utf-8')
diseases = file.readlines()
file.close()

idx_to_disease = {}

for i, element in enumerate(diseases):
    idx_to_disease[i] = element.strip()

model_lr, model_knn, model_dt, model_dl = get_model()

st.title("Health Buddy")

#### Style 1 Start ####

print('''
input_values = []

for i in range(MAX_SYMPTOMS):
    input_values.append(st.selectbox(f"Symptom {i + 1}", symptoms_list))

if st.button("Add Selectbox"):
    add_selectbox()

st.write("Selected Symptoms:")
for i, value in enumerate(input_values):
    st.write(f"{value}")
''')

#### Style 1 End ####

#### Style 2 Start ####

input_values = st.multiselect("Select symptoms:", symptoms_list, [], key="options")

if len(input_values) > MAX_SYMPTOMS:
    st.warning(f"Select up to {MAX_SYMPTOMS} symptoms. Please deselect some symptoms. Prediction might be inaccurate.")
    input_values = input_values[:MAX_SYMPTOMS]

#### Style 2 End ####

if st.button("Get Prediction"):
    show_prediction(input_values)