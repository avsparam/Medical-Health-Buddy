import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

df = pd.read_csv("Dataset/liver.csv")

df.dropna(inplace=True)

gender_dict = {'Male': 0, 'Female': 1}
df['Gender'] = [gender_dict[ele] for ele in df['Gender']]

skewed_cols = ['Albumin_and_Globulin_Ratio', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']
for c in skewed_cols:
    df[c] = df[c].apply('log1p')

minority = df[df.Dataset == 2]
majority = df[df.Dataset == 1]
minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])

df = pd.concat([pd.DataFrame(minority_upsample), pd.DataFrame(majority)], axis=0)

X = df.drop(columns=['Dataset', 'Direct_Bilirubin', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin'], axis=1)
Y = df['Dataset']

scaler = RobustScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = ExtraTreesClassifier()
scores = cross_validate(model, X_train, Y_train, cv=5, n_jobs=-1, verbose=1, return_estimator=True)

for est in scores['estimator']:
    accuracy = accuracy_score(Y_test, est.predict(X_test))
    print('Accuracy: ', round(accuracy * 100, 2))
