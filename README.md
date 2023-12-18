# Credit-card-Fraud-Detection
In this project, I detect fraudulent credit card transaction using machine learning and various python libraries. Our objective is to build a Fraud 
detection system using machine learning techniques. 

##  About Datasets
 Here, I use the datasets from (https://www.kaggle.com/datasets). It contains  total 284,807 transaction . Each transation is labelled either fraudulent or not 
 fraudulent. It contains only numerical input variables which are the result of a PCA transformation.  Features V1, V2, … V28 are the principal components obtained with PCA,
 the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction
 in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and 
 it takes value 1 in case of fraud and 0 otherwise.

#### 🛠 Technologies Used

    * Language: Python
    * IDE : Jupyter Notebook ,Visual studio
    * Analytical tools : (pandas,statstical method ,NUmpy) for
              data cleaning && (Matplotlib and seaborn )for EDA
    * Model BUilding : sklearn
    * Model Deployment: Streamlit
    
 ## Data Cleaning 
 In data cleaning section, i check for null value and duplicate value. i find there are 284807 rows and 31 columns where i found only 492 data are fraud.so we
 can say that it is imbalanced data sets.

 ## Exploratory Data Analysis

 In this section , i check statstical description of datasets. From datasets i seperate fraud and legit datasets. and then for that imbalanced datasets, i create  new datasets where
i perfrom random under sampling in legit datasets and take 492 legit datasets and combined with fraud datasets.
and find the mean of that new datasets and compare with previous datasets mean .There is not much difference in that mean which is good.

## Spliting dataset for train and test 
In this section , i split data into X and Y variable  and then perform train_test_split .

# Model Building
This datasets come under binary classification so i used logistic regression model. and use skelearn metrics for accuray score,precision score, recall and f1 score.
I get 92% accuracy for that model prediction.

## Model Deployment

for model deployment i used streamlit app .



 
 
