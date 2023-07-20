import streamlit as st
import predict

MAX_SYMPTOMS = 4

def show_prediction(input_values):
    predictions=predict.create_input_df(input_values)
    st.write(f'''# Predicted Disease using different models\n
    LR: {predictions[0]}\n
    KNN: {predictions[1]}\n
    DT: {predictions[2]}\n
    DL: {predictions[3]}''')


st.title("Health Buddy")
symptoms_list = predict.symptoms_list

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

