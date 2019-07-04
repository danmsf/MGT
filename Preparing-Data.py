#!/usr/bin/env python
# coding: utf-8

# In[239]:


import pandas as pd
import numpy as np
# import mysql.connector as mysql
import pyodbc
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy import signal
from scipy.interpolate import UnivariateSpline
# get_ipython().run_line_magic('matplotlib', 'inline')
import pylab
# matplotlib.rcParams['figure.figsize'] = (20.0, 5.0)
pylab.rcParams['figure.figsize'] = (20.0, 5.0)


# ### Connect to database

# In[240]:


db = pyodbc.connect('Driver={SQL Server};'
                    'UID=Dan;'
                    'Server=localhost;'
                    'Database=Cimplicity;'
                    'PWD=D@2019;'
                    'Trusted_Connection=no;')


# ### helper functions

# In[241]:


def filter_data(x, y, N=4, Wn=0.2):
    y.fillna(method='bfill', inplace=True)
    # Apply the filter
    B, A = signal.butter(N, Wn, output='ba')
    y_flt = signal.filtfilt(B,A,y)
    y_flt = pd.Series(y_flt,index =x, name ='y_flt' )
    return y_flt

def create_spline(X, y, slambda = 4,splinek = 1):
    # Spline Parameters
    x = X
    s = UnivariateSpline(x, y, s=slambda, k=splinek)
    y_hat = s(x)
    y_der = s.derivative()(x)
    return y_hat, y_der, s

def plotsigfilter(X,y,y_flt=None,spl=None, xlim=None, ylim=None, graphtype='r.'):
    fig=plt.figure(figsize=(20,5))
    #    filtname = title
    #    fig.suptitle(filtname, fontsize=14, fontweight='bold')
    ax =fig.add_subplot(1,1,1)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.plot(X,y,graphtype,ms=1)
    if y_flt is not None:
        ax.plot(X,y_flt,'b.',ms=1)
#     ax.plot(X,y_flt)
    if spl is not None:
        ax.plot(X, spl,'g-')


# ## Read in one sample

# In[242]:


# data = pd.read_sql("select BATCH_VAL0,stateNumber,nextState,Temp1,Temp2,VFD2, timeFromStartState ,timeFromStartBatch, timestamp from [dbo].[GSTAT_CYAN_CTR_Final] order by timestamp", db)
data = pd.read_sql("select * from [dbo].[GSTAT_CYAN_CTR_Final] where BATCH_VAL0=217631 order by timestamp", db)
od = pd.read_excel('C:\\Users\\Administrator\\Desktop\\OD.xlsx')
od.columns
data = pd.merge(data, od[['BATCH','MEAN', 'MEAN_LCL','MEAN_UCL']], left_on='BATCH_VAL0',right_on='BATCH')
data['ind_bad'] = ((data.MEAN > data.MEAN_UCL) |  (data.MEAN < data.MEAN_LCL))*1
test = data[data['BATCH']==217631]

# Set index to time from start batch
test.index = test.timeFromStartBatch
# drop duplicate timestamps
test = test[~test.index.duplicated()]
test.sort_index(inplace=True)


# In[243]:


test.columns


# In[244]:


len(test.columns)


# ## Data Filtering and Spline

# In[232]:


y_flt = filter_data(test.timeFromStartBatch, test.VFD2, N=4, Wn=0.02)
y_hat, y_der, s = create_spline(y_flt.index, y_flt, slambda = 4,splinek = 3)
plotsigfilter(test.index, test.VFD2, y_flt, spl=y_hat, xlim=[0,500])


# In[246]:


X = test[['Temp1', 'Temp2', 'Temp3', 'Temp5', 'Temp6', 'Temp7',
       'Temp8', 'Temp9', 'Temp10', 'Temp11', 'Temp12', 'RPM1', 'RPM2', 'RPM3',
       'Curr3', 'Flow1', 'Flow2', 'Flow3', 'Prss1', 'CV1', 'VFD2', 'CV3',
       'WT1', 'WT2', 'FIC_20COUT', 'FIC_20CPV', 'FIC_20CSP', 'IIC_01COUT',
       'IIC_01CPV', 'IIC_01CSP', 'PIC_20COUT', 'PIC_20CPV', 'PIC_20CSP',
       'TIC_01CMPV_1', 'TIC_01CMSP_1', 'TIC_01COUT_1', 'TIC_01CPV_1',
       'TIC_01CSP_1', 'TIC_01CSPV_1', 'TIC_01CSSP_1', 'TIC_01CMPV_2',
       'TIC_01CMSP_2', 'TIC_01COUT_2', 'TIC_01CPV_2', 'TIC_01CSP_2',
       'TIC_01CSPV_2', 'TIC_01CSSP_2']]


# In[249]:


len(X.columns)


# ## Spectograms of raw data

# In[251]:


frequency = 16000
nfft = 300
noverlap = None
fig, axs = plt.subplots(nrows=47, ncols=2, sharex=False, figsize=(20,150))
for i in range(47):
    ax = axs[i,0]
    ax.scatter(y=X.iloc[:,i], x=X.index)
    ax.set(ylabel = 'Variable {}'.format(i+1))
    ax = axs[i,1]
    powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(X.iloc[:,i], Fs=frequency, NFFT=nfft, noverlap = noverlap)


# ## PCA for dimensionality reduction

# In[252]:


# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
plt.figure(figsize=(20,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[230]:


frequency = 16000
nfft = 300
noverlap = None
fig, axs = plt.subplots(nrows=10, ncols=2, sharex=False, figsize=(20,40))
for i in range(10):
    ax = axs[i,0]
    ax.scatter(y=projected[:,i], x=X.index)
    ax.set(ylabel='Component {}'.format(i+1))
    ax = axs[i,1]
    powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(projected[:,i], Fs=frequency, NFFT=nfft, noverlap = noverlap)


# In[238]:


X_pca = pca.transform(X)
X_new = pca.inverse_transform(X_pca)
plt.figure(figsize=(20,5))
plt.scatter(X.TIC_01CMSP_2, X.TIC_01CPV_2, alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
