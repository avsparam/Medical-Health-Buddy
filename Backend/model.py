from decimal import Decimal
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

df_comb = pd.read_csv("..\Dataset\disease_symptoms_dataset.csv")

# creation of features and label for training the models
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

print(X.sum(axis=1).min())
print(X.sum(axis=1).mean())
print(X.sum(axis=1).median())
print(X.sum(axis=1).max())

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# lists used for accuracy plots
accuracy_list = []
cross_accuracy_list = []
model_list = []

# Multinomial NB Classifier
mnb = MultinomialNB()
mnb = mnb.fit(X, Y)
# prediction of labels for the test data
mnb_pred = mnb.predict(x_test)
# calculation of accuracy score based on predictions performed
# converting to Decimal as rounding with float is inaccurate
acc_mnb = round(Decimal(accuracy_score(y_test, mnb_pred) * 100), 2)
accuracy_list.append(acc_mnb)
model_list.append("MNB")
print(f"Accuracy (MNB) : {acc_mnb}%")

# Cross Validation Accuracy MNB
# performing cross validation with 5 different splits
scores_mnb = cross_val_score(mnb, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_mnb.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (MNB): {score}%")

# RF Classifier
rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
rf = rf.fit(X, Y)
# prediction of labels for the test data
rf_pred = rf.predict(x_test)
acc_rf = round(Decimal(accuracy_score(y_test, rf_pred) * 100), 2)
accuracy_list.append(acc_rf)
model_list.append("RF")
print(f"Accuracy (RF) : {acc_rf}%")

# Cross Validation Accuracy RF
# performing cross validation with 5 different splits
scores_rf = cross_val_score(rf, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_rf.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (RF): {score}%")

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=4)
knn = knn.fit(X, Y)
# prediction of labels for the test data
knn_pred = knn.predict(x_test)
acc_knn = round(Decimal(accuracy_score(y_test, knn_pred) * 100), 2)
accuracy_list.append(acc_knn)
model_list.append("KNN")
print(f"Accuracy (KNN) : {acc_knn}%")

# Cross Validation Accuracy KNN
# performing cross validation with 5 different splits
scores_knn = cross_val_score(knn, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_knn.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (KNN): {score}%")

# LR Classifier
lr = LogisticRegression()
lr = lr.fit(X, Y)
# prediction of labels for the test data
lr_pred = lr.predict(x_test)
acc_lr = round(Decimal(accuracy_score(y_test, lr_pred) * 100), 2)
accuracy_list.append(acc_lr)
model_list.append("LR")
print(f"Accuracy (LR) : {acc_lr}%")

# Cross Validation Accuracy LR
# performing cross validation with 5 different splits
scores_lr = cross_val_score(lr, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_lr.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (LR): {score}%")

# SVM Classifier
svm = SVC()
svm = svm.fit(X, Y)
# prediction of labels for the test data
svm_pred = svm.predict(x_test)
acc_svm = round(Decimal(accuracy_score(y_test, svm_pred) * 100), 2)
accuracy_list.append(acc_svm)
model_list.append("SVM")
print(f"Accuracy (SVM) : {acc_svm}%")

# Cross Validation Accuracy SVM
# performing cross validation with 5 different splits
scores_svm = cross_val_score(svm, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_svm.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (SVM): {score}%")

# DT Classifier
dt = DecisionTreeClassifier()
dt = dt.fit(X, Y)
# prediction of labels for the test data
dt_pred = dt.predict(x_test)
acc_dt = round(Decimal(accuracy_score(y_test, dt_pred) * 100), 2)
accuracy_list.append(acc_dt)
model_list.append("DT")
print(f"Accuracy (DT) : {acc_dt}%")

# Cross Validation Accuracy DT
# performing cross validation with 5 different splits
scores_dt = cross_val_score(dt, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_dt.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (DT): {score}%")

# MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(32, 32, 32), activation='relu', solver='adam', max_iter=50)
mlp = mlp.fit(X, Y)
# prediction of labels for the test data
mlp_pred = mlp.predict(x_test)
acc_mlp = round(Decimal(accuracy_score(y_test, mlp_pred) * 100), 2)
accuracy_list.append(acc_mlp)
model_list.append("MLP")
print(f"Accuracy (MLP) : {acc_mlp}%")

# Cross Validation Accuracy MLP
# performing cross validation with 5 different splits
scores_mlp = cross_val_score(mlp, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_mlp.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (MLP): {score}%")

# Voting Classifier
model1 = LogisticRegression()
model2 = SVC(probability=True)
model3 = RandomForestClassifier(n_estimators=10, criterion='entropy')

vc = VotingClassifier(estimators=[
    ('lr', model1),
    ('svc', model2),
    ('rf', model3)], voting='soft', verbose=True, n_jobs=-1)

vc.fit(x_train,y_train)
# prediction of labels for the test data
vc_pred = vc.predict(x_test)
acc_vc = round(Decimal(accuracy_score(y_test, vc_pred) * 100), 2)
accuracy_list.append(acc_vc)
model_list.append("VC")
print(f"Accuracy (VC) : {acc_vc}%")

# Cross Validation Accuracy DT
# performing cross validation with 5 different splits
scores_vc = cross_val_score(vc, X, Y, cv=5)
# mean of cross val score (accuracy)
score = round(Decimal(scores_vc.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (VC): {score}%")


# Convert symptom data to NumPy array
X_array = np.array(X, dtype=np.float32)
x_train_array = np.array(x_train, dtype=np.float32)
x_test_array = np.array(x_test, dtype=np.float32)

# One Hot Encode diseases data
encoder = OneHotEncoder()
Y_array = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
y_train_array = encoder.transform(np.array(y_train).reshape(-1, 1)).toarray()
y_test_array = encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()

# Define the model
tf_model = Sequential()

# Add layers to the model
tf_model.add(Dense(512, input_dim=X_array.shape[1])) # Input layer with 512 nodes and with a dimension of 489
tf_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))) # Hidden layer with 256 nodes
tf_model.add(Dropout(0.2))  # Dropout layer with a dropout rate of 0.2
tf_model.add(Dense(128, activation='relu')) # Hidden layer with 128 nodes
tf_model.add(Dense(64, activation='relu')) # Hidden layer with 64 nodes
tf_model.add(Dense(Y_array.shape[1], activation='softmax')) # Output layer

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True, use_ema=True)
tf_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

tf_model.summary()

# Train the model
tf_model.fit(X_array, Y_array, epochs=20, batch_size=64, validation_data=(x_test_array, y_test_array))

# Evaluate the model
loss, accuracy = tf_model.evaluate(x_test_array, y_test_array)
accuracy = round(Decimal(accuracy * 100), 2)
print(accuracy)
accuracy_list.append(accuracy)
model_list.append("DL")
print(f"Accuracy (DL): {accuracy}%")

kc_tf_model = KerasClassifier(build_fn=lambda:tf_model)
scores_dl = cross_val_score(kc_tf_model, X_array, Y_array, cv=5)
score = round(Decimal(scores_dl.mean() * 100), 2)
cross_accuracy_list.append(score)
print(f"Cross Validation Accuracy (DL): {score}%")