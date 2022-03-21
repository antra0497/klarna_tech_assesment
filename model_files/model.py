import pandas as pd
import numpy as np
from numpy import mean

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def drop_nan_col(data):
  '''
  This function takes a dataframe as input and drops the columns
  which has more than 40% of the data as NaN. It returns the new dataframe.
  '''
  row_count =data.shape[0] #total row count
  threshold = row_count*0.40 #defining threshold of 40% of total size
  counter =0 #counter flag

  for col in data:
    if col == 'default': #skipping the default column
      pass
    elif data[col].isnull().sum() > threshold : #dropping the column if has NaN more than threshold
      data= data.drop(columns=col, axis=1)
      counter +=1
  l ='{} columns were droped, they had more than 40% data missing, New shape of data is {}'.format(counter, data.shape)
  return data, l #returning the modified df and log

def fill_nan(data):
  '''
  This function takes dataframe as an input and 
  fill zero in place of NAN and returns the modified dataframe.
  '''
  count =0
  for col in data.columns:
    if col == 'default': #skipping the default column
      pass
    elif data[col].isnull().sum()!=0: #checking if column has any NAN values
      data[col] = data[col].fillna(0) #fill those NAN with zero
      count +=1
  l='NaN values in {} columns filled with zero'.format(count)   
  return data, l #returning the modified df and log

def convert_c2n(data):
  '''
  This function takes data frame as an input and converts 
  all the categorical or boolean features into numerical features,
  and returns the modified dataframe
  Here we are using Ordinal Encoder to execute the task.
  '''
  counter =0
  encoder = OrdinalEncoder() #initiallizing the encoder
  for col in data.columns:
    if data[col].dtype == 'O' or data[col].dtype == 'bool': #checking for object or boolean data type
      data[col]= encoder.fit_transform(data[col].values.reshape(-1,1))
      counter +=1
  l ='{} columns converted from Categorical/Boolean to Numerical '.format(counter)
  return data, l #returning the modified df and log

def standard_scaling(data, test_flag = False):
  '''
  This function is used for normalising the data. 
  We use Standard scaler to to fit tranform the data 
  on the scale of [0,1] using z-score. 
  '''
  print(data.shape) 
  x_std = StandardScaler()
  # Standardizing the features, to bring all the featues on a scale between [0,1]
  x= x_std.fit_transform(data)
    
  x =pd.DataFrame(x)
  return x

def dimensionality_reduction(data):
  '''
  This function takes a data frame as input and apply PCA for dimentionality reduction
  and returns the components which has high variance and will have more feature importance 
  In this case we have initialized PCA for 95% variance.
  '''
  #print(type(data))
  pca = PCA(n_components=26) # with variance of 95%
  principalComponents = pca.fit_transform(data)
  v ='Explained Variance is {}'.format(pca.explained_variance_)
  nc ='Count of PCA components: {}'.format(pca.n_components_)
  return principalComponents, v, nc #returning the modified df and log

def data_prep(data):
  '''
  Funtion: Data Cleaning Pipeline

  Input: Raw Test Data
  Output: Processed Test Data
  Processing: Dropping columns, Changing column type, Filling Null values,
              Sandard Scaling, Dimentionality reduction
  '''

  log=[] # list to maintain extra details as log
  log.append('Original shape of Test data: {}'.format(data.shape))

  log.append('DATA PROCESSING')
  #Step1 : Dropping columns with 40% values as NAN
  df, _ = drop_nan_col(data)
  log.append(_)

  #Step 2: Changing categorical to numerical columns
  df_c, _ = convert_c2n(df)
  log.append(_)  

  #Step 3: Filling null values with zero
  df_n, _ = fill_nan(df_c)
  log.append(_)  

  #Step 4: Data Normalization
  df_norm = standard_scaling(df_n, test_flag=True)

  #Step 5: Feature reduction
  df_reduced, v, nc= dimensionality_reduction(df_norm)
  log.append('DIMENTIONALITY REDUCTION')
  log.append(nc)
  log.append(v)
  
  #returning processed test dataframe
  return df_reduced, log

def predict_data(config, model):
  '''
  It takes the raw test data form main.py, 
  and returns the predicted values with other details as log   
  '''
  #changing the received file intp pandas dataframe
  df = pd.DataFrame(config)
  
  #Calling Data Cleaning Pipeline
  df_processed, log = data_prep(df)

  #Getting results
  y_pred = model.predict(df_processed)

  #returning the results
  return y_pred, log


  