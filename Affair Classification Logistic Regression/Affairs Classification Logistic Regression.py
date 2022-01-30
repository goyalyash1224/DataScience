# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 10:28:57 2022

@author: yash
"""


'''

perform Classification using logistic regression 
and check your model accuracy using confusion matrix 
and also through .score() function.

NOTE: Perform OneHotEncoding for occupation and occupation_husb, 
since they should be treated as categorical variables. 

Careful from dummy variable trap for both!!


Percentage of total women actually had an affair.

(note that Increases in marriage rating and religiousness correspond 
to a decrease in the likelihood of having an affair.)

 Predict the probability of an affair for a random woman 
 not present in the dataset. 
 She's a 25-year-old teacher who graduated college, 
 has been married for 3 years, 
 has 1 child, 
 rates herself as strongly religious, 
 rates her marriage as fair, 
 and her husband is a farmer.
 
 
 
 '''
 
#import basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
data  = pd.read_csv('affairs.csv')
 
data.head()
data.columns
 
data.describe()
data.isnull().any() 


#Devide data into features and labels according to problem
features = data.iloc[:,:-1]
labels = data.iloc[:,-1] 

'''


np.unique(features['occupation'].values)
np.unique(features['occupation_husb'].values)

features.occupation.value_counts()
features.occupation_husb.value_counts()


'''

#In the features, two column 'occupation', 'occupation_husb' are defined as
#categorical data those need to be convert in numeric values using OneHotEncoder
#for oneHotEncoding, two classes need to be import

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cTransformer1 = ColumnTransformer([('endcode',OneHotEncoder(),[-1])],remainder='passthrough')
cTransformer2 = ColumnTransformer([('endcode',OneHotEncoder(),[-1])],remainder='passthrough')

#apply column transformer on last column
features_re = cTransformer1.fit_transform(features)
print(features_re.shape) 

#Remove dummy variable for reduce redundancy from the features
features_re = features_re[:,1:]
print(features_re.shape)

#Again apply column transformer on last columm
features_re = cTransformer2.fit_transform(features_re)
print(features_re.shape) 

 #Remove dummy variable for reduce redundancy from the features
features_re = features_re[:,1:]
print(features_re.shape)


#percentage of women actually had an affair
count = labels.value_counts()
print("Percentage of women actually had an affair is :" , round((count[0]/count.sum()*100),2))



#Now split the data in test and train
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features_re,labels, random_state=43, test_size = 0.2)


#Perform classification using LogisticRegression 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(features_train,labels_train) 

pred = classifier.predict(features_test)

#confusion matrix
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(pred,labels_test)) 
print(classifier.score(features_test, labels_test))
print(classifier.score(features_train,labels_train)) 

#we got 74% test accuracy using logisitic without modification

'''
 
  Predict the probability of an affair for a random woman 
 not present in the dataset. 
 She's a 25-year-old teacher who graduated college, 
 has been married for 3 years, 
 has 1 child, 
 rates herself as strongly religious, 
 rates her marriage as fair, 
 and her husband is a farmer.
 
 '''
features.columns
w1 = [3,25,3,1,4,16,2,2]
w1_re = cTransformer1.transform([w1])
w1_re = w1_re[:,1:]
print(w1_re.shape)
w1_re = cTransformer2.transform(w1_re)
w1_re = w1_re[:,1:]
print(w1_re.shape)

print("Women has affair :  " + ("Yes" if classifier.predict(w1_re)==1 else "No"))
#predition is No here, so this women should not have an affair