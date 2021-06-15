# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:54:01 2021

@author: nikhil.barua
"""
#Exploring the dataset


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('kyphosis.csv')

df.head()

df.info()

sns.pairplot(df,hue='Kyphosis')

#traiing and testing the dataset

from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))


#If the data is big Random forest will usually do better than Decision Tree as the data gets bigger and bigger

df['Kyphosis'].value_counts()