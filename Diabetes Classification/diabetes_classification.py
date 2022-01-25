# -*- coding: utf-8 -*-
"""
Problem Statement:


The datasets consists of several medical predictor variables and one target variable, Outcome. 
Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

perform classification and build a machine learning model by performing standard scaling to accurately predict whether or not the patients in the 
dataset have diabetes or not?

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data and analyse the data
data = pd.read_csv('diabetes.csv')

data.shape
data.head()

data.columns

#Analysis of two columns 'skin thickness' and 'Insulin' cause these two have more zeros, less valuable to our model
data['SkinThickness'].value_counts()
data['Insulin'].value_counts()
#pie plot
data['SkinThickness'].value_counts().plot(kind='pie')
data['Insulin'].value_counts().plot(kind='pie')


#Seeing these two pie charts, we can drop these two columns from the data
data_re = data.drop(['SkinThickness','Insulin'], axis = 1)



data_re.shape
data_re[data_re['BloodPressure']==0].shape



# using replace method, we replace zeros in two columns('BMI', 'BloodPressure') with mean values
mean_BMI = data_re['BMI'].mean()
mean_bp = data_re['BloodPressure'].mean()
data_re['BMI'].replace(0,mean_BMI,inplace = True)
data_re['BloodPressure'].replace(0,mean_bp,inplace = True)

#find if, there is any missing values
data_re.isnull().any(axis=0)

#seperate features and labels
features = data_re.iloc[:,0:6].values
labels = data_re.iloc[:,-1].values



#feature scaling using standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)



#split the data in train or test
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 76)

#KNN algorithm for classification
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, p =2, metric = 'minkowski')
classifier.fit(features_train,labels_train)
labels_pred = classifier.predict(features_test)


#making the confusion matrix to see the accuracy of our model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(labels_pred,labels_test)
print(accuracy_score(labels_pred,labels_test),"\n",cm)
print(accuracy_score(labels_train, classifier.predict(features_train)))
#we got 77% test accuracy and 84% train accuracy. so we conclue it as a decent model.

