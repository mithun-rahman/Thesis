import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

series = pd.read_csv('thesis.csv',index_col='date',parse_dates=True)

series['orders'] = series['orders']

xf = series
mm = series
xx = series
xx.head(5)

plt.figure(figsize=(16, 5))
plt.plot(series)
plt.show()

# fit model
model = ARIMA(series, order=(1,0,4))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals

# split into train and test sets
X = series.values
size = int(len(X) * 0.85)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions = list()
# walk-forward validation
smape = 0
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,4))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	smape += (abs((obs-yhat)/obs))
	
from sklearn.metrics import mean_absolute_error as MAE

plt.plot(test)
plt.plot(predictions, color='red', linestyle = 'dotted')
plt.plot(predictions, color='red', linestyle = 'dashed')
plt.figure(figsize=(2, 20))
plt.show()


plt.plot(test)
plt.plot(predictions, color='red', linestyle = 'dotted')
plt.plot(predictions, color='red', linestyle = 'dashed')
plt.figure(figsize=(2, 20))
plt.show()


predictions1 = list()
# walk-forward validation
smape = 0
for t in range(len(test)):
	model1 = ARIMA(history, order=(1,0,0))
	model_fit1 = model1.fit()
	output = model_fit1.forecast()
	yhat = output[0]
	predictions1.append(yhat)
	obs = test[t]
	history.append(obs)
	smape += (abs((obs-yhat)/obs))
	
plt.plot(test, color='red')
plt.plot(predictions1, color='blue', linestyle = 'dashed')
plt.figure(figsize=(2, 20))
plt.show()

xx = mm
xx.head(5)

from scipy.stats import norm
import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(xx['orders'],order=(2, 1, 0),seasonal_order=(1,1,0,52))
results=model.fit()

xx['forecast']=results.predict(start=133,end=157,dynamic=True)

xx[['orders','forecast']].plot(figsize=(12,8))

yy = xx[133:]
plt.plot(yy)

xx = yy['forecast']
print(xx)

plt.plot(xx)

plt.plot(test)
plt.plot(predictions, color='red',linestyle = 'dotted')
plt.plot(predictions1, color='blue', linestyle = 'dashed')
plt.figure(figsize=(2, 20))

plt.show()

print(xx.head(5))
plt.plot(test)
plt.plot(predictions, color='red',linestyle = 'dotted')
plt.plot(predictions1, color='blue', linestyle = 'dashed')
plt.figure(figsize=(4, 30))

zz = pd.Series(predictions)  # zz for ARIMA
yy = pd.Series(predictions1) # yy AR

tt=zz.astype(float)
tt.plot()
kk=yy.astype(float)
kk.plot()

plt.figure(figsize=(2, 20))

tt = pd.DataFrame(test) 
kk = pd.DataFrame(xx)
yy = pd.DataFrame(yy)
zz = pd.DataFrame(zz)

ax = tt.plot()
yy=yy.astype(float)
yy.plot(ax=ax)
zz=zz.astype(float)
zz.plot(ax=ax)

sarima = list()

for x in xx:
  sarima.append(x)
  
xx = pd.DataFrame(sarima)   # xx = SARIMA
ax = tt.plot(label='Observed')            # tt = test
yy=yy.astype(float)
yy.plot(ax=ax)            # yy = AR
zz=zz.astype(float)       
zz.plot(ax=ax)            # zz = ARIMA
xx.plot(ax=ax)            # xx = SARIMA


plt.figure(figsize=(12, 8))
ax = tt.plot(label='Observed')            # tt = test

yy.plot(ax=ax)            # yy = AR
zz=zz.astype(float)       
zz.plot(ax=ax)            # zz = ARIMA
xx.plot(ax=ax)            # xx = SARIMA

plt.figure(figsize=(12, 8))
plt.plot(tt, label="line1")
plt.plot(xx)
plt.plot(yy)
plt.plot(zz)

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams["figure.autolayout"] = True
line1, = plt.plot(tt ,color='blue', marker = '^', label="observed")
err=1000
line2, = plt.plot(yy, linestyle = 'dashed', label="AR")
line2, = plt.plot(zz, color='green', label="ARIMA")
line2, = plt.plot(xx, marker = 'o', label="SARIMA")
leg = plt.legend(loc='upper left')
plt.show()

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams["figure.autolayout"] = True
line1, = plt.plot(tt ,color='blue', marker = '^', label="observed")
#line2, = plt.plot(yy, linestyle = 'dashed', label="AR")
line2, = plt.plot(zz, color='green', marker = 'o', label="ARIMA")
#line2, = plt.plot(xx, marker = 'o', label="SARIMA")
leg = plt.legend(loc='upper left')
plt.show()

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams["figure.autolayout"] = True
line1, = plt.plot(tt ,color='blue', marker = '^', label="observed")
#line2, = plt.plot(yy, linestyle = 'dashed', label="AR")
#line2, = plt.plot(zz, color='green', marker = 'o', label="ARIMA")
line2, = plt.plot(xx, marker = 'o', color='red', label="SARIMA")
leg = plt.legend(loc='upper left')
plt.show()

print("Mean Absolute Error {}".format((MAE(xx, test)/err)))


print("Root Mean Squared Error {}" .format(sqrt(mean_squared_error(xx,test))/err))# SARIMA

print("Mean Absolute Error {}".format((MAE(yy, test)/err)))


print("Root Mean Squared Error {}" .format(sqrt(mean_squared_error(yy,test))/err))# ARIMA

print("Mean Absolute Error {}".format((MAE(zz, test)/err)))


print("Root Mean Squared Error {}" .format(sqrt(mean_squared_error(zz,test))/err))# AR

