# klarna_tech_assesment


# Defaulter Analysis: Imbalance Class Problem

This is a simple flask application to do prediction for Credit Defaulter:

**Steps:** 

    1. Upload test file as csv
    2. Get the prediction and other logs
    3. Download the results

**Note: minimum requirement for test data are as follows:**

    1. Atleast 50 rows of test data
    2. .csv file format only

# Model Info.
Following models were trained and tested on validation data, and best model was selected for the app.

**Model 1: Logistic Regression**

    1. without class weights
    2. with class weigths 
    3. modified with grid search
    
**Model 2: Decision Trees and Ensemble models:**

    1. Ada Boost
    2. XGBoost
    3. Random Forest 

**Model 3: Neural Network, keras Sequential API**

    1. without class weights
    2. with class weigths
    3. over sampling 

For more details refer to [Jupyter File](https://github.com/antra0497/klarna_tech_assesment/blob/main/klarna_assignment.ipynb) 

# Application is deployed on Heroku:

Link: https://klarna-pred.herokuapp.com/

![image](https://user-images.githubusercontent.com/25953832/159197639-d91b386e-10a4-4352-8b4a-5b4f7a20b578.png)
