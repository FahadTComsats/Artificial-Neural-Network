#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:41:22 2019

@author: fahadtariq
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#Handling categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing keras libraries and pacakages

import keras
from keras.models import Sequential 
from keras.layers import Dense

#Initializing the ANN

classifier = Sequential()

#ADDING THE INPUT LAYER AND FIRST INPUT LAYER

classifier.add(Dense(units = 6,kernel_initializer='uniform',activation="relu",input_dim = 11)) #First Hidden layer is added

#Second Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer='uniform',activation="relu"))

#Adding the Output layer
classifier.add(Dense(units = 1,kernel_initializer='uniform',activation="sigmoid"))

#Compiling the ANN      type of sokistic graditent decent

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit the model into trainig set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

