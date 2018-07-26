# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:43:25 2018

@author: saash
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('SLA_ML_models_2.csv')
dataset.head()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 13].values
X
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)



plt.plot(y_pred, color = 'red')
plt.plot(y_test, color ='blue')
plt.show()

plt.scatter(y_test,y_pred)

from sklearn import metrics
metrics.explained_variance_score(y_test,y_pred)

sns.distplot ((y_test-y_pred),bins =50)