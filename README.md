# Air Pressure System Failure Detection And Cost Minimization

## Introduction:
The goal of this project is to develop a predictive model that accurately detects failures in the Air Pressure System (APS) of heavy Scania trucks. The APS is responsible for generating pressurized air used in various functions of the truck, such as braking and gear changes. By detecting failures in the APS system, we can prevent potential breakdowns and minimize the associated costs. The dataset used for this project contains anonymized operational data collected from Scania trucks, with a focus on failures related to the APS component.

## Dataset Overview:
The dataset consists of two main parts: a training set and a test set. The training set contains 60,000 examples, with 59,000 belonging to the negative class (trucks with failures not related to APS) and 1,000 belonging to the positive class (trucks with APS component failures). The test set contains 16,000 examples. Each record in the dataset has 171 attributes, including both single numerical counters and histograms with bins representing different conditions. The attribute names have been anonymized for proprietary reasons, and missing values are denoted by "na." The dataset has been carefully selected by experts to represent real-world scenarios.

## Objective:
The main objective of this project is to develop a predictive model that can accurately classify trucks as either positive (APS component failure) or negative (non-APS component failure) based on the given dataset. Additionally, the model should aim to minimize the cost associated with failures by accurately identifying instances with type 1 and type 2 failures. The total cost of the prediction model is calculated as the sum of Cost_1 multiplied by the number of instances with type 1 failure and Cost_2 multiplied by the number of instances with type 2 failure.

## Approach:
To achieve the project's objectives, the following steps will be undertaken:

1.Exploratory Data Analysis (EDA):

* Perform an initial analysis of the dataset to understand its structure, attribute types, and distribution.
* Examine the relationship between attributes and identify any correlations with APS component failures.
* Handle missing values appropriately, either by imputation or elimination, based on the dataset's characteristics.
* Utilize the **IterativeImputer** from the sklearn.impute module to perform iterative imputation and estimate missing values.

2.Feature Engineering and Selection:

* Preprocess the dataset by transforming and normalizing the data as necessary.
* Extract meaningful features from the available attributes to enhance the model's performance.
* Select relevant features that have the most impact on the prediction task, using techniques like feature importance or dimensionality reduction methods.
* Utilize the **RFE (Recursive Feature Elimination)** class from the sklearn.feature_selection module to recursively select important features for the prediction task.

3.Model Development:

* Split the preprocessed dataset into training and validation subsets.
* Train various machine learning models, such as logistic regression, decision trees, random forests, or gradient boosting algorithms.
* Train a **RandomForestClassifier** from the sklearn.ensemble module, which is an ensemble learning method that uses a collection of decision trees to make predictions.
* Utilize appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score, to assess model performance.
* Optimize the models by tuning hyperparameters using techniques like grid search or random search.

4.Cost Optimization:

* Incorporate the cost matrix into the model evaluation process to minimize the total cost of failures.
* Adjust the classification thresholds based on the cost matrix to balance the trade-off between different types of failures.
* Select the model that achieves the best balance between accurate prediction and cost minimization.

5.Model Evaluation:

* Evaluate the final model's performance on the test set to ensure generalization and reliability.
* Generate relevant performance metrics and interpret the results.
* Prepare a comprehensive report documenting the methodology, findings, and recommendations.
* If deemed successful, deploy the model in a production environment for real-time predictions.

## Conclusion:
By developing a predictive model that accurately detects APS component failures in Scania trucks, we can help minimize breakdowns and associated costs. The project involves exploring the dataset

## Results
### Accuracy: 99.46 among top 1% on kaggle on this dataset
