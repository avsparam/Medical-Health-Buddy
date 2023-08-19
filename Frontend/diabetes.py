import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

diabetes_dataset = pd.read_csv('..\Dataset\diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)

scaler = StandardScaler()
scaler.fit(X)
std_data = scaler.transform(input_data_reshaped)

classifier = pickle.load(open('..\Models\diabetes_model.pkl', "rb"))
prediction = classifier.predict(std_data)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')