import pandas as pd
import pickle
from tensorflow import keras
import numpy as np

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
    predictions.append(idx_to_disease[int(np.argmax(prediction_dl))])

    return predictions


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

pickle_in = open("Models/model_lr.pkl","rb")
model_lr = pickle.load(pickle_in)

pickle_in = open("Models/model_knn.pkl","rb")
model_knn = pickle.load(pickle_in)

pickle_in = open("Models/model_dt.pkl","rb")
model_dt = pickle.load(pickle_in)

model_dl = keras.models.load_model("Models/model_dl.keras")

#input_symptoms = ['abscess', 'shortness breath', 'testicular pain', 'vomiting']
#print(create_input_df(input_symptoms))