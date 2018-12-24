#Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:, 2:4].values
y = dataset.iloc[:,4].values

#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Logistic Regression in Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predicting
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
  