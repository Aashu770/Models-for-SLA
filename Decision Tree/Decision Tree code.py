# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:44:48 2018

@author: saash
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('SLA_ML_models_2.csv')
dataset.head()
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)



df1=pd.DataFrame(X_test)
df2 = pd.DataFrame(y_test)
df3= pd.DataFrame(y_pred)
df4=pd.concat([df1,df2,df3],axis=1)

writer =pd.ExcelWriter('Decision Tree output.xlsx')
df4.to_excel(writer,'Sheet1')
writer.save()