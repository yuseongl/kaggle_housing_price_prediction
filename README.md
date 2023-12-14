# kaggle_housing_price_prediction
Repository for my personal project to housing price prediction using deep neural network

# Model
This ANN model has perceptron of multilayer fully connected and composed of 4-layer 
including 1-input, 3-hidden and 1-fully connected layer(Do not confuse it with 
the fully connected layer of DNN architecture, this is a layer combine all nodes)

# Evaluation
For evaluation used 5-fold cross validation and used mse, mae, r2score validation metric

# Preprocessing
This model is just pipeline for using machine learning, so composed minimal structure to run.
It is preprocessing architecture that:
- first handling missing value of target
- dividing type of data as numeric and category 
- filling missing value of numeric data as statistics(In this model used min value of each column)
- encoding categorical data using one-hot encoder
- merging each type data as dataframe
- matching train data set columns to test data set columns

# Visualization
You can visualization change of loss value and learning-rate using 'loss' module.
if you want other function add in this module

# Util
1. If you want resampling about imbalance of inpute-feauture, use 'resample' module in 'dataset' module.
2. you can use other loss fuction maked in 'loss' module including rmse, rmsle loss function
    * don't recommend using this loss functoin due to differential value goes to denominator