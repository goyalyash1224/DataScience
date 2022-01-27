# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:43:00 2022

@author: yash
"""
"""

Problem statement:

Predict the salary using "Salary.csv" dataset. Also include the backward elimination to 
eliminate features with least importance.

Print the RMSE of linear regression.

"""



#import necessary libraries
import pandas as pd
import numpy as np

#load the data and do basic data analysis
data = pd.read_csv('Salary.csv')

data.head()
data.shape
data.describe()
data.isnull().any()

#Here we have first column as catagorical data 
#It need to convert to numerical data using OneHotEncoder and ColumnTransformer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#object for column Transformer
cTransform = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
data_re = cTransform.fit_transform(data)

#Here OneHotEndoder geneartes one redundany column so that need to be drop
data_re = data_re[:,1:]

#devide data into features and lables
features = data_re[:,:-1]
labels = data_re[:,-1]







'''

#Now for linear regression and backward elimination we use OLS algorithm from statesmodels.api
import statsmodels.api as sm

#by adding constant to features , we will get best fit line according to algorithm
features = sm.add_constant(features[:,:])

#OLS regressor 
regressor_ols = sm.OLS(endog= labels,exog = features)
#fit the regressor on features to get p_values
model_ols = regressor_ols.fit()

p_values = model_ols.pvalues
print(model_ols.summary())
print(p_values)


'''
#Now to find best optimal features in data, p value of the column should be greater
#than 5%
#Automation of elimination of least important column one by one
#Importance of a feature is inversely propotional to P value of that feature


'''
Wrong approach as optimal_feature_array get updated all the time, but what should be done is:
    [1,2,3,4,5] -> [1,2,4,5] -> [1,2,4] -> [1,4] ->[1]



while(max(p_values)>0.05):
    p_max = max(p_values)
    feature_index = np.where(p_values == p_max)
    optimal_feature_array = [i for i in range(len(p_values)) if (i != feature_index[0])]
    model_ols = sm.OLS(endog= labels,exog = features[:,optimal_feature_array]).fit()
    p_values = model_ols.pvalues
    print(p_values)


print(optimal_feature_array)



Right approach is to delete that feature column that is not needed
'''

#Now for linear regression and backward elimination we use OLS algorithm from statesmodels.api
import statsmodels.api as sm


features_optimal = features[:,[0,1,2,3,4]]

while(True):
    regressor_ols = sm.OLS(endog = labels, exog= features_optimal).fit()
    p_values = regressor_ols.pvalues
    if p_values.max() > 0.05:
        features_optimal = np.delete(features_optimal,p_values.argmax(),1)
    else:
        break
    
print(features_optimal.shape)


#Coparitive study of using features and optimal features
from sklearn.linear_model import LinearRegression

regressor_normal = LinearRegression()
regressor_optimal = LinearRegression()

regressor_normal.fit(features,labels)
regressor_optimal.fit(features_optimal,labels)

label_pred_normal = regressor_normal.predict(features)
label_pred_optimal = regressor_optimal.predict(features_optimal)

#print RMSE for both model to compare
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(labels,label_pred_normal)))

print(np.sqrt(metrics.mean_squared_error(labels,label_pred_optimal)))
    
'''
5345.698942183195
5402.465424087967

Here we can see the performance of both models are similar but we only use 2 features 
in optimal_regressor. Hence we can reduce the features using this feature_selection technique.

'''

    
    
    