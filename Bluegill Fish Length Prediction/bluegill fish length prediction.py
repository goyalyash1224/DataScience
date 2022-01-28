# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:07:33 2022

@author: yash
"""



"""
In 1981, 78 bluegills were randomly sampled from Lake Mary in Minnesota. 
The researchers (Cook and Weisberg, 1999) measured and recorded the 
following data:
(Import bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish

Potential Predictor (Independent Variable): age (in years) of the fish.


How is the length of a bluegill fish best related to its age? 
(Linear/Quadratic nature?)

What is the length of a randomly selected five-year-old bluegill fish? 
Perform polynomial regression on the dataset.


"""

#important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Read the data and do basic analysis
data = pd.read_csv('bluegills.csv')
data.head()
data.describe()

data.isnull().any()  #No missing values

#Plot scatter plot 
plt.scatter(data['age'],data['length'])
plt.plot(data['age'],data['length'])

features = data.iloc[:,0:1].values
labels = data.iloc[:,-1].values

#We can see age and length are not linear corelated so we use polinomial regression 
#to predict the length using age

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

features_poly = PolynomialFeatures(degree=3)
features_high_degree = features_poly.fit_transform(features)

regressor = LinearRegression()

regressor.fit(features_high_degree,labels)


#plot predicted data (in red) and scatter plot of label vs features in green. 
# we can see model performance 
plt.plot(features,regressor.predict(features_high_degree),color = 'red',label='predicted length')
plt.scatter(features,labels,color='green',label='Given length')
plt.xlabel('age')
plt.ylabel('length')
plt.title('comparison between predicted and given data')
plt.legend()




#What is the length of a randomly selected five-year-old bluegill fish? 
print('length of 5 year bluegill fish is: ', regressor.predict(features_poly.fit_transform([[5]]))[0])



















