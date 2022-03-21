# klarna_tech_assesment


# Defaulter Analysis: Imbalance Class Problem

Delployed a simple flask application to do prediction for Credit Defaulter:

**Steps to use the Application:** 

    1. Upload test file as csv
    2. Get the prediction and other logs
    3. Download the results

**Note: minimum requirement for test data are as follows:**

    1. Atleast 50 rows of test data
    2. .csv file format only

# Model Info.

This case study was the classic example of imbalance class classification problem in machine learning where data for one class is far more than another class, here data for defaulter was present in 1:99 ratio. In common ML classification approach, algorithm assumes equal weights for all the classes and hence will results in low accuracy. Classifier Level Method such as Cost sensitive modelling is the workaround for such situations, in which we explicitly tell the model to assign more weights to minority classes. Another method is trying Data Level Methods like over sampling, where we try to make equal distribution of all the classes in mini batches, there by sampling more data from minority class to that of majority class.

> Following models were trained and tested on validation data, and best model was selected for the app.

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
    
> **Model Evaluation:** Since this is an imbalance class problem, accuracy of the model is not the correct metrics to look at as it will always be biased towards majority class, hence we are suggested to look into F1 score, Precision Recall and ROCAUC. In this case study we will be focusing on False Negatives as this might cost a lot of money in real life if the model flags defaulter as non-defaulter. This also backs up the rule of thumb which is very straightforward: the higher the value of the ROC AUC metric, the better the model performance. Our next focus should be True Negatives. but having said that a slightly higher False Positives is fine as there is no harm in being extra cautious.


> For more details refer to [Jupyter File](https://github.com/antra0497/klarna_tech_assesment/blob/main/klarna_assignment.ipynb) 

# Application is deployed on Heroku:

Link: https://klarna-pred.herokuapp.com/

![image](https://user-images.githubusercontent.com/25953832/159197639-d91b386e-10a4-4352-8b4a-5b4f7a20b578.png)
