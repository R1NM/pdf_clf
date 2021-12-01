# -*- coding: utf-8 -*-
from sklearn import svm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

#탐지 data
parsed_name="pre.csv"
parsed=pd.read_csv(parsed_name,dtype='float64',header=None)

clf=joblib.load('trained_svm.pkl')
scaler=joblib.load('trained_scaler.pkl')
parsed=scaler.transform(parsed)
pre=clf.predict(parsed)

#print prediction
for i in range(len(pre)):
	if pre[i]==0:
		print("file",i,": malicious")
	else:
		print("file",i,":","benign")
#
