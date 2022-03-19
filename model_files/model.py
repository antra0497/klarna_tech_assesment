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
  print('{} columns droped and size is {}'.format(counter, data.shape))
  return data

def fill_nan(data):
    for col in data.columns:
        if col == 'default': #skipping the default column
            pass
        elif data[col].isnull().sum()!=0:
            data[col] = data[col].fillna(0)
    print(data.shape)      
    return data

def convert_c2n(data):
  counter =0
  encoder = OrdinalEncoder()
  for col in data.columns:
    if data[col].dtype == 'O' or data[col].dtype == 'bool':
      data[col]= encoder.fit_transform(data[col].values.reshape(-1,1))
      counter +=1
  print('{} columns converted to numerical '.format(counter))
  print(data.shape) 
  return data  

def standard_scaling(data, test_flag = False):

    print(data.shape) 
    x_std = StandardScaler()
    # Standardizing the features, to bring all the featues on a scale between [0,1]
    x= x_std.fit_transform(data)
    
    x =pd.DataFrame(x)
    print(x.shape)

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

    return principalComponents

def data_prep(data):

  df = drop_nan_col(data)

  df_c = convert_c2n(df)

  df_n = fill_nan(df_c)

  df_norm = standard_scaling(df_n, test_flag=True)

  df_reduced = dimensionality_reduction(df_norm)

  return df_reduced

def predict_data(config, model):

    df = pd.DataFrame(config)

    df_processed = data_prep(df)
    y_pred = model.predict(df_processed)

    return y_pred


  