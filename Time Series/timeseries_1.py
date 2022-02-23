# After preprocess the data we move to forecasting 


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn import preprocessing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly as py
from plotly import tools
from plotly.offline import iplot
from plotly.subplots import make_subplots
import seaborn as sns
sns.set()
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import TimeDistributed

from keras.initializers import he_normal, he_uniform
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import statsmodels.api as sm
import time
import warnings
import itertools


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

pk = series
pk.shape

pk.plot(figsize=(16,9))

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pandas as pd

# Multiplicative Decomposition 
mul_result = seasonal_decompose(pk['orders'], model='multiplicative')

# Additive Decomposition
add_result = seasonal_decompose(pk['orders'], model='additive')

# Plot
plt.rcParams.update({'figure.figsize': (8,8)})
mul_result.plot().suptitle('\nMultiplicative Decompose', fontsize=12)

add_result.plot().suptitle('\nAdditive Decompose', fontsize=12)
plt.show()

#For checking stationarity of series we use adfuller test
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(pk.orders)

# Differerencing the data
pk['orders']=np.log(pk.orders)
pk['orders']=pk['orders'].ewm(alpha=0.5).mean()

adf_test(pk.orders)

plt.plot(pk.orders)
#ACF Plot, from here we get q value

sm.graphics.tsa.plot_acf(pk.orders);

#PACF Plot, from here we get p value

sm.graphics.tsa.plot_pacf(pk.orders);

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(pk.orders, order=(1,0,4))
model_fit = model.fit()
print(model_fit.summary())

from math import sqrt
from sklearn.metrics import mean_squared_error
plt.plot(pk.orders,color='r')
plt.plot(model_fit.predict(dynamic=False),color='g')
plt.legend(['Actual', 'Predicted'])

# Checking model results
model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()

# Seasonal - fit stepwise auto-ARIMA
import pmdarima as pm
smodel = pm.auto_arima(pk.orders, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=52,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()


