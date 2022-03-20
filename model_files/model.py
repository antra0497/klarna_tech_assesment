import pandas as pd
import numpy as np
from numpy import mean

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def drop_nan_col(data):
  row_count =data.shape[0] #total row count
  threshold = row_count*0.40 #defining threshold of 40% of total size
  counter =0

  for col in data:
    if col == 'default': #skipping the default column
      pass
    elif data[col].isnull().sum() > threshold : #dropping the column if has NaN more than threshold
      data= data.drop(columns=col, axis=1)
      counter +=1
  l ='{} columns were droped, they had more than 40% data missing, New shape of data is {}'.format(counter, data.shape)
  return data, l

def fill_nan(data):
  count =0
  for col in data.columns:
    if col == 'default': #skipping the default column
      pass
    elif data[col].isnull().sum()!=0:
      data[col] = data[col].fillna(0)
      count +=1
  l='NaN values in {} columns filled with zero'.format(count)   
  return data, l

def convert_c2n(data):
  counter =0
  encoder = OrdinalEncoder()
  for col in data.columns:
    if data[col].dtype == 'O' or data[col].dtype == 'bool':
      data[col]= encoder.fit_transform(data[col].values.reshape(-1,1))
      counter +=1
  l ='{} columns converted from Categorical/Boolean to Numerical '.format(counter)
  return data, l

def standard_scaling(data, test_flag = False):

    print(data.shape) 
    x_std = StandardScaler()
    # Standardizing the features, to bring all the featues on a scale between [0,1]
    x= x_std.fit_transform(data)
    
    x =pd.DataFrame(x)
    return x

def dimensionality_reduction(data):

    print(type(data))
    pca = PCA(n_components=26) # with variance of 95%
    principalComponents = pca.fit_transform(data)

    print('Explained Variance: ', pca.explained_variance_)
    print('\n')
    print('Count of PCA components: ', pca.n_components_)
    print('\n')
    print('Visulating the variance vs no.of components')

    v ='Explained Variance is {}'.format(pca.explained_variance_)
    nc ='Count of PCA components: {}'.format(pca.n_components_)
    return principalComponents, v, nc

def data_prep(data):
  log=['.',] # list to maintain log
  log.append('Original shape of data: {}'.format(data.shape))

  log.append('DATA PROCESSING')
  df, _ = drop_nan_col(data)
  log.append(_)
  df_c, _ = convert_c2n(df)
  log.append(_)  
  df_n, _ = fill_nan(df_c)
  log.append(_)  
  df_norm = standard_scaling(df_n, test_flag=True)
  df_reduced, v, nc= dimensionality_reduction(df_norm)
  log.append('DIMENTIONALITY REDUCTION')
  log.append(nc)
  log.append(v)
  
  return df_reduced, log

def predict_data(config, model):

    df = pd.DataFrame(config)

    df_processed, log = data_prep(df)
    y_pred = model.predict(df_processed)

    return y_pred, log


  