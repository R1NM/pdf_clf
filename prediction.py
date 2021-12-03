# -*- coding: utf-8 -*-
from sklearn import svm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

#prediction data
parsed_name="test.csv"
parsed=pd.read_csv(parsed_name,header=None)
pre_Y=parsed.loc[:,0]
pre_Y=pd.Series.tolist(pre_Y)
for i in range(len(pre_Y)):
  if pre_Y[i]=='M':
    pre_Y[i]=0
  else:
    pre_Y[i]=1
pre_X=parsed.loc[:,1:]
pre_X=pre_X.astype('float64')

#load model
clf=joblib.load('trained_svm.pkl')
scaler=joblib.load('trained_scaler.pkl')
pre_X=scaler.transform(pre_X)

#predict
pre_score=clf.score(pre_X,pre_Y)
print("prediction score:",pre_score)
pre=clf.predict(pre_X)
