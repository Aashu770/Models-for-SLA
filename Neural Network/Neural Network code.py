# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:50:28 2018

@author: saash
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('SLA_ML_models.csv')
dataset.head()

X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

y=y.reshape(-1,1)

onehotencoder = OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.08, random_state = 123)

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(8, input_dim = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(8, kernel_initializer= 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(5, kernel_initializer= 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 30, nb_epoch = 100)

y_pred = classifier.predict(X_test)

df1=pd.DataFrame(X_test)
df2 = pd.DataFrame(y_test)
df3= pd.DataFrame(y_pred)
df4=pd.concat([df1,df2,df3],axis=1)

'''X_test
df[13]= y_test[0]
df[14]=y_pred'''

writer =pd.ExcelWriter('Neutal Network output.xlsx')
df4.to_excel(writer,'Sheet1')
writer.save()









