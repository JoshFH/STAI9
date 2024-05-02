import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

## Reading the dataset
FurnitureData = pd.read_csv("Furniture Price Prediction.csv")
FurnitureData.shape
# Removing duplicate rows
FurnitureData = FurnitureData.drop_duplicates()
FurnitureData.shape
# Displaying a section of data
FurnitureData.head(10)

## Visualising the distribution of the Target variable
Distribution = FurnitureData['price']
fig = plt.figure(figsize = (8,8))
distGraph = fig.gca()
# Sets range of values for each column (smaller value more detail)
bin_width = 100 
num_bins = int((Distribution.max() - Distribution.min()) / bin_width)
# Title and labels of the graph 
distGraph.set_title('Price Distribution of Furniture')
distGraph.set_xlabel('Price')
distGraph.set_ylabel('Frequency')
Distribution.hist(ax=distGraph, bins=num_bins)
# Increasing tick interval for graph readability
tick_interval = 5000
min_price = np.floor(Distribution.min() / tick_interval) * tick_interval
max_price = np.ceil(Distribution.max() / tick_interval) * tick_interval
ticks = np.arange(min_price, max_price + tick_interval, tick_interval)
plt.xticks(ticks)

# Removing values larger than 25000
FurnitureData = FurnitureData.drop(FurnitureData[FurnitureData['price'] > 25000].index)
# Listing details of FurnitureData after removing outlier values
#FurnitureData.info()
# Reprinting distribution graph based on new values
Distribution = FurnitureData['price']
fig = plt.figure(figsize = (8,8))
distGraph = fig.gca()
bin_width = 100 
num_bins = int((Distribution.max() - Distribution.min()) / bin_width)
distGraph.set_title('Price Distribution of Furniture')
distGraph.set_xlabel('Price')
distGraph.set_ylabel('Frequency')
Distribution.hist(ax=distGraph, bins=num_bins)
tick_interval = 5000
min_price = np.floor(Distribution.min() / tick_interval) * tick_interval
max_price = np.ceil(Distribution.max() / tick_interval) * tick_interval
ticks = np.arange(min_price, max_price + tick_interval, tick_interval)
plt.xticks(ticks)

## Data Exploration
FurnitureData.head()
FurnitureData.tail()
#FurnitureData.info()
FurnitureData.describe(include='all')
FurnitureData.nunique

## Visual Data Analysis 
# For cataergorical (type) data
Furniture_Type = FurnitureData['type'].value_counts()
top_20_types = Furniture_Type.head(20)
fig, ax = plt.subplots(figsize=(8, 8))
typeGraph = ax.bar(x=top_20_types.index, height=top_20_types.values)
# calls the index of the top20types for the x label, and has height set to the max value
# Adding labels to each bar
for bar in typeGraph:
    height = bar.get_height()
    ax.annotate(f'{height}',  xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    # xy coordinates are placed at the top of the bar and in the middle 
    # horizontal alingment is centered, vertical allignment is bottom
ax.set_title('Furniture Type Counts')
ax.set_xlabel('Furniture Type')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')

# rate distribution
rateDistribution = FurnitureData['rate']
fig = plt.figure(figsize = (8,8))
rateGraph = fig.gca()
num_bins = 20
rateGraph.set_title('Rating Distribution of Furniture')
rateGraph.set_xlabel('Rating')
rateGraph.set_ylabel('Frequency')
rateDistribution.hist(ax=rateGraph, bins=num_bins)

# delivery distribution
delivDistribution = FurnitureData['delivery']
fig = plt.figure(figsize = (8,8))
delivGraph = fig.gca()
num_bins = 50
delivGraph.set_title('Delievery Cost Distribution of Furniture')
delivGraph.set_xlabel('Delivery Cost')
delivGraph.set_ylabel('Frequency')
delivDistribution.hist(ax=delivGraph, bins=num_bins)

# sale distribution
# converting sale to a float value
FurnitureData['sale'] = FurnitureData['sale'].str.replace('%', '')
FurnitureData['sale'] = FurnitureData['sale'].astype(int)

saleDistribution = FurnitureData['sale']
fig = plt.figure(figsize = (10, 6))
saleGraph = fig.gca()

saleGraph.set_title('Discount Percentage Distribution of Furniture')
saleGraph.set_xlabel('Discount Percent')
saleGraph.set_ylabel('Frequency')
saleDistribution.hist(ax=saleGraph, bins=20)

## Remove outliers
FurnitureData = FurnitureData.drop(FurnitureData[FurnitureData['delivery'] > 2000].index)
# Reprint delivery graph with updated data
delivDistribution = FurnitureData['delivery']
fig = plt.figure(figsize = (8,8))
delivGraph = fig.gca()
num_bins = 50
delivGraph.set_title('Delievery Cost Distribution of Furniture')
delivGraph.set_xlabel('Delivery Cost')
delivGraph.set_ylabel('Frequency')
delivDistribution.hist(ax=delivGraph, bins=num_bins)

## Null values
FurnitureData.isnull().sum()
FurnitureData = FurnitureData.dropna(subset=['price'])
FurnitureData.isnull().sum()
#FurnitureData.info()

## Visual Correlations
predictors = ['rate', 'sale', 'delivery']
for predictor in predictors:
    FurnitureData.plot.scatter(x=predictor, y='price', figsize=(10,5), title=predictor+" VS "+ 'price')

## Statistical correlation
# Creating the correlation matrix
variables = ['price', 'rate', 'sale', 'delivery']
CorrelationData = FurnitureData[variables].corr()
priceCorrelationData = CorrelationData['price']

# Correlation for rating (removing zero rating entries)
RatedFurnitureData = FurnitureData.drop(FurnitureData[FurnitureData['rate'] == 0].index)
RatedFurnitureData.plot.scatter(x='rate', y='price', figsize=(10,5), title="rate VS price")
RatedVariables = ['price', 'rate']
RatedCorrelation = RatedFurnitureData[RatedVariables].corr()

# Data selected for ML
SelectedVariables = ['price','sale','delivery']
MLData = FurnitureData[SelectedVariables]
#MLData.info()

#Saving data subset for deployment
MLData.to_pickle('MLData.pkl')

#Delete comment to display all graphs used on run 
#plt.show()

## Machine Learning Development 
TargetVariable = 'price'
predictors = ['sale', 'delivery']

# Separate Target Variable and Predictor Variables
X=MLData[predictors].values
y=MLData[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
#K value is 4 so therefore dataset is split into quarters (1876/4 = 469)
#Therefore 1/4 is for testing and 3/4 (1407 entries) is for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)
# Sanity check for the sampled data
X_train.shape
y_train.shape
X_test.shape
y_test.shape
 
## Multiple Regression Algorithms
###########################################################################
# Linear Regression
from sklearn.linear_model import LinearRegression
RegModel = LinearRegression()

# Print all the parameters of Linear regression
#print(f"\n{RegModel}")

#Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)
prediction=LREG.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
    TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])
#MAPE = Mean Absolute Percentage Error
MAPE=np.mean(TestingDataResults['APE'])
#Median Mean Absolute Percentage Error
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100 - MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
##print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

###########################################################################
# Decision Trees (Multiple if-else statements!)
from sklearn.tree import DecisionTreeRegressor
RegModel = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')
# Good Range of Max_depth = 2 to 20

# Printing all the parameters of Decision Tree
#print(f"\n{RegModel}")

# Creating the model on Training Data
DT=RegModel.fit(X_train,y_train)
prediction=DT.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)))

# Plotting the feature importance of columns
feature_importances = pd.Series(DT.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Importance of Features')
plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
plt.xlim(0, 1.0)

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
##print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Ran into issues with the Graphviz module, Ignoring visualising of Decision Tree
'''
# Visualing Decision Tree
from IPython.display import Image
from sklearn import tree
import pydotplus

TargetVariable = ['price']

# Create DOT data
dot_data = tree.export_graphviz(RegModel, out_file=None, feature_names=predictors, class_names=TargetVariable)
# printing the rules
#print(dot_data)
# Draw graph
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
# Show graph
#.write_png('tree.png')
'''

###########################################################################
# Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestRegressor
RegModel = RandomForestRegressor(max_depth=6, n_estimators=150,criterion='friedman_mse')
# Good range for max_depth: 2-10 and n_estimators: 100-1000

# Printing all the parameters of Random Forest
#print(f'\n{RegModel}')

# Creating the model on Training Data
RF=RegModel.fit(X_train,y_train)
prediction=RF.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))

# Plotting the feature importance for columns
feature_importances = pd.Series(RF.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Importance of Features')
plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
plt.xlim(0, 1.0)

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
##print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Ran into issues with the Graphviz module, Ignoring visualising of Decision Tree
'''
# Plotting a single Decision Tree from Random Forest

# Create DOT data for the 6th Decision Tree in Random Forest
dot_data = tree.export_graphviz(RegModel.estimators_[5] , out_file=None, feature_names=predictors, class_names=TargetVariable)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png(), width=2000,height=2000)
'''

###########################################################################
# Adaboost (Boosting Multiple Decision Trees)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Choosing Decision Tree with 6 level as the weak learner
DTR=DecisionTreeRegressor(max_depth=3)
RegModel = AdaBoostRegressor(n_estimators=100, learning_rate=0.04)

# Printing all the parameters of Adaboost
#print(f'\n{RegModel}')

# Creating the model on Training Data
AB=RegModel.fit(X_train,y_train)
prediction=AB.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, AB.predict(X_train)))

# Plotting the feature importance for columns
feature_importances = pd.Series(AB.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Importance of Features')
plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
plt.xlim(0, 1.0)

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
##print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

###########################################################################
# Xtreme Gradient Boosting (XGBoost)
from xgboost import XGBRegressor
RegModel=XGBRegressor(max_depth=2,
                      learning_rate=0.1,
                      n_estimators=1000,
                      objective='reg:linear',
                      booster='gbtree')

# Printing all the parameters of XGBoost
#print(f'\n{RegModel}')

# Creating the model on Training Data
XGB=RegModel.fit(X_train,y_train)
prediction=XGB.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, XGB.predict(X_train)))

# Plotting the feature importance for columns
feature_importances = pd.Series(RF.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Importance of Features')
plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
plt.xlim(0, 1.0)

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])


MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
##print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Issue with Graphiz module and plotting trees 
'''
#Plotting a single Decision tree out of XGBoost
from xgboost import plot_tree
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(XGB, num_trees=10, ax=ax)
plt.show()
'''

###########################################################################
#kNN
# K-Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsRegressor
RegModel = KNeighborsRegressor(n_neighbors=3)

# Printing all the parameters of KNN
#print(f'\n{RegModel}')

# Creating the model on Training Data
KNN=RegModel.fit(X_train,y_train)
prediction=KNN.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
#print('R2 Value:',metrics.r2_score(y_train, KNN.predict(X_train)))

# Plotting the feature importance for columns
# The variable importance chart is not available for KNN

#print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
TestingDataResults.head()

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['price']-TestingDataResults['Predictedprice']))/TestingDataResults['price'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
#print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
#print('Median Accuracy on test data:', MedianAccuracy)

# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
#print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

#Used for displaying all plotted graphs
#plt.show()

## Deployment
# Reconfirming use of KNN
RegModel = KNeighborsRegressor(n_neighbors=3)
# Fitting to 100% of the data
Final_KNN_Model=RegModel.fit(X,y)

# Save model as a serialized file 
import pickle
import os

# Saving the Python objects as serialized files can be done using pickle library
# Here let us save the Final model
with open('Final_KNN_Model.pkl', 'wb') as fileWriteStream:
    pickle.dump(Final_KNN_Model, fileWriteStream)
    # Don't forget to close the filestream!
    fileWriteStream.close()

print('pickle file of Predictive Model is saved at Location:',os.getcwd())

from re import IGNORECASE

# Building function for the model
def FunctionPredictResult(InputData):
    Num_Inputs=InputData.shape[0]

    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input

    # Appending the new data with the Training data
    MLData=pd.read_pickle('MLData.pkl')
    #InputData=InputData.append(DataForML, ignore_index=True)
    InputData = pd.concat([InputData, MLData], ignore_index=True)

    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)

    # Maintaining the same order of columns as it was during the model training
    predictors = ['sale', 'delivery']

    # Generating the input values to the model
    X=InputData[predictors].values[0:Num_Inputs]

    # Loading the Function from pickle file
    import pickle
    with open('Final_KNN_Model.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()

    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    print(PredictionResult)

# Call function with new sample data
NewSampleData=pd.DataFrame(data=[[8,860],[24,627]],columns=['sale', 'delivery'])
print(NewSampleData)

# Calling the Function for prediction
FunctionPredictResult(InputData=NewSampleData)

# Updating model to integrate with an API (Takes Input)
def FunctionGeneratePrediction(inp_sale, inp_delivery):

    # Creating a data frame for the model input
    SampleInputData=pd.DataFrame(
     data=[[inp_sale, inp_delivery]],
     columns=['sale', 'delivery'])

    # Calling the function defined above using the input parameters
    Predictions=FunctionPredictResult(InputData= SampleInputData)

    # Returning the predictions
    #print(Predictions.to_json())

# Function call
FunctionGeneratePrediction(inp_sale=8, inp_delivery=860)

# Web Deployment
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/prediction_api', methods=["GET"])
def prediction_api():
    try:
        # Getting the paramters from API call
        sale_value = float(request.args.get('sale'))
        delivery_value=float(request.args.get('delivery'))

        # Calling the funtion to get predictions
        prediction_from_api=FunctionGeneratePrediction(inp_sale = sale_value, inp_delivery = delivery_value)
        return (prediction_from_api)

    except Exception as e:
        return('Something is not right!:'+str(e))
    
# Starting the API engine
if __name__ =="__main__":

    # Hosting the API in localhost
    app.run(host='127.0.0.1', port=9000, threaded=True, debug=True, use_reloader=False)
    # Interrupt kernel to stop the API