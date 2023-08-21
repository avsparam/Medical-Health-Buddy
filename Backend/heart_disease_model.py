import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

heart_data = pd.read_csv('Dataset/heart_disease_data.csv')

X = heart_data.drop(columns='target', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy: ', test_data_accuracy)

pickle.dump(model, open("Models/heart_disease_model.pkl", 'wb'))
