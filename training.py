# -*- coding: utf-8 -*-

from sklearn import svm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

#training dat file
csv_name="output.csv"

#load data
data=pd.read_csv(csv_name,header=None)
Y=data.loc[:,0]
Y=pd.Series.tolist(Y)
for i in range(len(Y)):
  if Y[i]=='M':
    Y[i]=0
  else:
    Y[i]=1
X=data.loc[:,1:]
X=X.astype('float64')

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#scaled svm
scaler = StandardScaler()
scaler.fit(x_train)
train_x_scaled =scaler.transform(x_train)
test_x_scaled = scaler.transform(x_test)

clf = svm.SVC(C=100,gamma=0.001)
clf.fit(train_x_scaled, y_train)
score=clf.score(test_x_scaled, y_test)
print("model accuracy:",score)

#pre=clf.predict(test_x_scaled)
#print(pre)

joblib.dump(clf, 'trained_svm.pkl')

joblib.dump(scaler,'trained_scaler.pkl')

#result
print("saved trained svm model")
