# Fraud Detection Model: Project Overview

* **Goal**: Create a model that helps to identify fraudulent credit card transactions.
* Normalized unscaled data to match previously scaled features.
* Used Random Under-Sampling to deal with unbalanced dataset.
* Optimized Logistic Regression classifier using Recursive Feature Elimination in Cross-Validation (RFECV) and GridSearchCV to reach the best model.


# Outline
## Dataset
*The original data can be downloaded [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)*

The dataset contains transactions made by credit cards in September 2013, over two days, by European cardholders.

There are 284,807 transactions in the dataset, of which 492 (0.17%) were considered to be frauds (i.e., 'Class' = 1), meaning that the dataset is **extremely unbalanced**.

## Scaling
A PCA was performed in the original dataset and the majority of the dataset features ("V" features) are the obtained Principal Components of that transformation.

Columns 'Time' and 'Amount' were scaled (normalized) to match the scale of the other columns.

## Unbalanced dataset and Random Under-Sampling
The large amount of identified legitimate transactions versus frauds suggests that this is an unbalanced dataset.

In order to avoid common problems related with this type of dataset, random under-sample technique was applied by generating a subsample with a 50/50 ratio (i.e., same amount) of fraud and legitimate transactions.

## Logistic Regression Classifier
Performed feature selection via Recursive Feature Elimination in Cross-Validation (RFECV).

The new balanced subsample was split into training and testing subsets (80/20 split).

Determined the parameters that give the best predictive cross-validation score using GridSearchCV.

Evaluated Logistic Regression model based on accuracy score, classification report and confusion matrix.

## Improvements and Next Steps
Presented several points of improvement to be further explored in this project:
* Apply other subsampling technique (e.g., SMOTE);
* Under/over-sampling during cross-validation;
* Deeper data pre-processing;
* Testing other classifiers.

### Resources
The Jupyter Notebook (fraud_detection.ipynb) was developed using [Google Colaboratory](http://colab.research.google.com/).
