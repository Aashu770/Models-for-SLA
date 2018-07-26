# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:00:00 2018

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
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



plt.plot(y_pred, color = 'red')
plt.plot(y_test, color ='blue')
plt.show()

'''plt.scatter(y_test,y_pred)'''

'''sns.distplot ((y_test-y_pred),bins =50)'''


from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test,y_pred))
print (confusion_matrix (y_test,y_pred))

df=pd.DataFrame(X_test)
df[13]= y_test
df[14]=y_pred

writer =pd.ExcelWriter('Kneighbor output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
