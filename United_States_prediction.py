# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 00:14:21 2020

@author: anmol
"""

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from pmdarima import auto_arima
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")
 
# Loading the dataset and look at datatyes and shape of data
data = pd.read_csv('total_cases.csv')
print (data.head())
print ('\n Data Types:')
print (data.dtypes)

# reading data with datetime date set as index
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('total_cases.csv', parse_dates=['date'], index_col='date',date_parser=dateparse)
print ('\n Parsed Data:')
print (data.head())

# make a new dataframe of United States country
df = data[["United States"]]
df.dropna(inplace=True)

plt.plot(df)

# Find errors
def timeseries_evaluation_metrics_func(y_true, y_pred):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MSE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    
# this function to check timeseries is stationary or not using Dickey-Fuller test
def Augmented_Dickey_Fuller_Test_func(series , column_name):
    print (f'Results of Dickey-Fuller Test for column: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:====>")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")
        

Augmented_Dickey_Fuller_Test_func(df['United States'],['United States'])

# Split data in train and test
X = df[['United States']]
train, test = X[0:217], X[216:]

# Apply model, pmdarima module will help to find p, q, d 
stepwise_model = auto_arima(train,start_p=1, start_q=1,
    max_p=7, max_q=7, seasonal=False,
    d=None, trace=True,error_action='ignore',suppress_warnings=True, stepwise=True)

stepwise_model.summary()

# forecasting for next 60 days to train
forecast,conf_int = stepwise_model.predict(n_periods=60,return_conf_int=True)
forecast = pd.DataFrame(forecast, columns=['US_pred'])

# making days which store dates of next 30 days 
start = datetime.strptime('2020-08-03', '%Y-%m-%d')
end = datetime.strptime('2020-10-01', '%Y-%m-%d')
days = pd.date_range(start, end)

# make dataframe to store limit of predicted output
df_conf = pd.DataFrame(conf_int,columns= ['Upper_bound','Lower_bound'])
df_conf["new_index"] = days
df_conf = df_conf.set_index("new_index")

# Evaluting the errors
timeseries_evaluation_metrics_func(test, forecast[0:30])

forecast["new_index"] = days
forecast = forecast.set_index("new_index")

# Final plot of Covid-19 predictions
plt.rcParams["figure.figsize"] = [15,7]
plt.plot(train, label='Train ')
plt.plot(test, label='Test ')
plt.plot(forecast, label='Predicted ')
plt.plot(df_conf['Upper_bound'], label='Confidence Interval Upper bound ')
plt.plot(df_conf['Lower_bound'], label='Confidence Interval Lower bound ')
plt.title('US_RMSE: %.4f'% np.sqrt(metrics.mean_squared_error(test, forecast[0:30])))
plt.legend(loc='best')
plt.ylabel('Total Cases')
plt.xlabel('Date')
plt.show()