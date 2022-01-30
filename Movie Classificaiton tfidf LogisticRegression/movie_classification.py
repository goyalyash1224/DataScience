# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 01:42:22 2022

@author: yash goyal
"""

"""
Import movie.csv file

There are two categories: 
Pos (reviews that express a positive or favorable 
sentiment)and 
Neg (reviews that express a negative or 
unfavorable sentiment).

For this code challenge, we will assume that 
all reviews are either positive or negative;
there are no neutral reviews.

Perform sentiment analysis on the text reviews 
to determine whether its positive
or negative and build confusion matrix to determine 
the accuracy.

"""


#import the libraries
import pandas as pd
import numpy as np




#data analysis
data = pd.read_csv('movie.csv')
data.head()
data.isnull().any()

#no missing value in data



#labels(class) are in two class, pos and neg, need to convert in numeric format
#labelEncoder will do exactly same
from sklearn.preprocessing import LabelEncoder
labels = LabelEncoder().fit_transform(data['class'])


#split the data into train test
from sklearn.model_selection import train_test_split

features_train, features_test , labels_train, labels_test = train_test_split(data['text'],labels, test_size=0.2, random_state= 34)


#conver text data into vectors using tfIdf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5).fit(features_train)
print(len(vect.get_feature_names_out()))

features_train_vectorized = vect.transform(features_train)





#fit the logistic regression on the data for classification
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(features_train_vectorized,labels_train)

predictions = model.predict(vect.transform(features_test))



#print confusion matrix and roc_auc_score
from sklearn.metrics import confusion_matrix, roc_auc_score
print(confusion_matrix(labels_test, predictions))
print(roc_auc_score(labels_test, predictions))

#we got 83% AUC score using logistic regression classification with tfidf vectorizer.
