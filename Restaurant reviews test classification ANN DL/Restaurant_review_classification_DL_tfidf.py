# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 00:22:21 2022

@author: yash
"""

 #import the important librabris
import pickle
import time
from time import sleep
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import os
from sklearn.model_selection import train_test_split
#import libraries for ANN
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import Adadelta,Adam,RMSprop
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adadelta,Adam,RMSprop
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

#Defining deep learning model
def ann_model(input_s):
    #Deep Layer Model building in Keras
    
    model = Sequential()
    model.add(Dense(700,input_dim=input_s))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def data_analysis(data_path):
    global features,target
    data = pd.read_csv(str(data_path)+"//Restaurant_Reviews.tsv",sep='\t')
    data = pd.read_csv("Restaurant_Reviews.tsv",sep='\t')
    print(data.head())
    print(data.describe)
    print("Shape of the data: ",data.shape)
    print("Any column has NULL value : ", data.isnull().any())
    #drop the column having the null value
    # data.dropna(inplace= True)
    # print("Columns dropped....")
    print("Shape of the data: ",data.shape)
    print(data.columns)
    

    #split the data into target and features 
    print("Splitting data into target and features")
    target = data['Liked']
    print('shape of target data and type :', target.shape,type(target))
    features = data['Review']
    print('shape of features data and type :', features.shape,type(features))    
    
    #convert text data into vectors using tfIdfVectorizer
    vectorizer = TfidfVectorizer(min_df=5)
    vectorized_features = vectorizer.fit_transform(features)
    return vectorized_features,target   
    

#defining the main function         
def main():
    global features,target
    data_path = os.getcwd()
    features,target = data_analysis(data_path)
    print("total number of features: ", features.shape[1])
    model = ann_model(features.shape[1])
    print("type of features and target: ",type(features),type(target))
    print(model)
   
    #split the data into train and test data using train_test_split
    features_train,features_test, labels_train, labels_test = train_test_split(features.toarray(),target,random_state=43,test_size=.3)
    #fit the model on  data
    
    model.fit(features_train,labels_train,batch_size=10,epochs=50)
    #predit the classes for features test
    labels_pred = model.predict(features_test)
    print("Type of labels_pred.......",type(labels_pred),labels_pred.shape)
    
    labels_pred = labels_pred.flatten()
    print("Type of labels_pred.......",type(labels_pred),labels_pred.shape)
    print(labels_pred)
    labels_pred = labels_pred > 0.5
    # labels_pred = (labels_pred > 0.5)
    #print zip file of predicted labels and test labels
    # print(list(zip(labels_test, labels_pred)))
    print(confusion_matrix(labels_test, labels_pred))
    print(accuracy_score(labels_test, labels_pred))
    print("Model trained.....")
    
    pickle.dump(model, open('model.pkl','wb'))
    print("model dumped in pickle file")
    print("Program stopped")     



    
# Calling the main function 
if __name__ == '__main__':
    main()



    