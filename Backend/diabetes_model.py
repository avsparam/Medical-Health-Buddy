import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

diabetes_dataset = pd.read_csv('Dataset/diabetes_new.csv')

X = diabetes_dataset.drop(columns=['CLASS', 'No_Pation', 'ID'], axis=1)
Y = diabetes_dataset['CLASS']

gender_dict = {'M': 0, 'F': 1}
for i, ele in enumerate(X['Gender']):
    X['Gender'][i] = gender_dict[ele.strip().upper()]

class_dict = {'Y': 0, 'P': 1, 'N': 2}
for i, ele in enumerate(Y):
    Y[i] = class_dict[ele.strip().upper()]

scaler = StandardScaler()
X = scaler.fit_transform(X)
Y=Y.astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy: ', test_data_accuracy)

pickle.dump(classifier, open("Models/diabetes_model_new.pkl", 'wb'))
