#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
time_data=pd.read_csv("C:/Users/kushal/Downloads/hcl.csv")
time_data.head(10)


# In[3]:


hcl_data=time_data.dropna()
hcl_data.shape


# In[4]:


hcl_data.index=pd.to_datetime(hcl_data.Date)
hcl_data


# In[5]:


hcl_data_agl=hcl_data['Prev Close']['2013-01-01':'2013-12-02']
hcl_data_agl.shape


# In[6]:


#one year data of entries,beware of holidays as well
hcl_data_agl.describe()


# In[7]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
ax1=fig.add_subplot(111)
ax1.plot(hcl_data_agl)
ax1.set_xlabel("Time Frame")
ax1.set_ylabel("Stock price for hcltech")


# In[8]:


rollmean=hcl_data_agl.rolling(12).mean()
rollstd=hcl_data_agl.rolling(12).std()


# In[9]:


plt.figure(figsize=(16,7))
fig=plt.figure(1)
orig=plt.plot(hcl_data_agl,color='blue',label='Original')
mean=plt.plot(rollmean,color='red',label='Rolling Mean')
std=plt.plot(rollstd,color='black',label='Rolling Std')
plt.legend(loc='best')
plt.show()


# In[10]:


#for a time series model to be constant over time, the mean and std should be constant
# from the above series plot std is constant but the rolling mean isn't constant
#adfueller test also can be used to test if the series is stationary.


# In[11]:


#making series stationary
import numpy as np
plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log=np.log(hcl_data_agl)
plt.plot(ts_log)


# In[12]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(ts_log,freq=1,model='mutiplicative')
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.figure(figsize=(16,7))
fig=plt.figure(1)
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.show()


# In[13]:


#trying out the differencing time series 
plt.figure(figsize=(16,7))
fig=plt.figure(1)
ts_log_diff=ts_log-ts_log.shift(1)
rolmean=ts_log_diff.rolling(12).mean()
rolstd=ts_log_diff.rolling(12).std()
org=plt.plot(ts_log_diff,color='red',label='Org')
mean=plt.plot(rolmean,color='blue',label='mean')
std=plt.plot(rolstd,color='black',label='std')
plt.legend(loc='best')
plt.show()


# In[14]:


# we can see that there is no upward pattern for mean and standard deviation


# In[15]:


# this forecast looks stationary-> you canperform adfueller test to check
# we can plot acf and pacf 


# In[16]:


from statsmodels.tsa.arima_model import ARIMA
plt.figure(figsize=(16,8))
model=ARIMA(ts_log,order=(2,1,2))
results_arima=model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_arima.fittedvalues,color='red')


# In[17]:


#blue line shows the actual graph, the red line shows the forecasted plot line
# taking back values to the original scale
arima_diff_predictions=pd.Series(results_arima.fittedvalues,copy=True)
print(arima_diff_predictions)


# In[18]:


arima_diff_predictions_cumsum=arima_diff_predictions.cumsum()
print(arima_diff_predictions_cumsum.head())


arima_log_prediction=pd.Series(ts_log,index=ts_log.index)
arima_log_prediction=arima_log_prediction.add(arima_diff_predictions_cumsum,fill_value=0)
print(arima_log_prediction)


# In[19]:


plt.figure(figsize=(12,8))
predictions_arima_model=np.exp(arima_log_prediction)
plt.plot(hcl_data_agl)
plt.plot(predictions_arima_model)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_arima_model-hcl_data_agl)**2)/len(hcl_data_agl)))


# In[26]:



import pmdarima as pm
def arimamodel(timeseries):
    automodel=pm.auto_arima(timeseries,seasonal=True,trace=True)
    return automodel

arimamodel(ts_log)


# In[34]:


results_arima.plot_predict(1,264)


# In[ ]:




