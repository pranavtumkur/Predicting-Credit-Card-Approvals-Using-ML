# Predicting-Credit-Card-Approvals-Using-ML

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this project, I have tried to model an automatic credit card approval predictor using machine learning techniques, just like the real banks do!

<img width="497" alt="credit-card-account-rejection-and-cancellation-rates-creeping-up" src="https://user-images.githubusercontent.com/65482013/85938716-54776880-b92d-11ea-9a6c-b0872dae514a.png">

The dataset used in this project is the [Credit Card Approval dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval) from the UCI Machine Learning Repository.

## Data Features

The features of this dataset have been anonymized to protect the privacy, but this [blog](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) gives us a pretty good overview of the probable features. The probable features in a typical credit card application are Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output. The dataset therefore, has a mixture of numerical and non-numerical features which will require some preprocessing.

## Preprocessing

Our dataset contains both numeric and non-numeric data (specifically data that are of float64, int64 and object types). The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Finally, the dataset has missing values, which are labeled with '?', which can be seen in the last cell.

For numeric columns, we impute the missing values with a strategy called mean imputation. For non-numeric columns the mean imputation strategy would not work. This needs a different treatment. We are going to impute these missing values with the most frequent values as present in the respective columns. This is [good practice](https://www.datacamp.com/community/tutorials/categorical-data) when it comes to imputing missing values for categorical data in general.

Finally, we need to convert categorical and non-categorical features because many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called [label encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

## Training the model and predicting

Essentially, predicting if a credit card application will be approved or not is a classification task. According to UCI, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved.

This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.

Which model should we pick? A question to ask is: are the features that affect the credit card approval decision process correlated with each other? We use a correlation technique to see if the parameters and find out a high degree of correlation. Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases like a Logistic Regression model.

We will now evaluate our model on the test set with respect to [classification accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy). But we will also take a look the model's [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/). In the case of predicting credit card applications, it is equally important to see if our machine learning model is able to predict the approval status of the applications as denied that originally got denied. If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. The confusion matrix helps us to view our model's performance from these aspects.

## Grid searching and making the model perform better

Our model was pretty good! It was able to yield an accuracy score of almost 84%.

For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.

Let's see if we can do better. We can perform a [grid search](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/) of the model parameters to improve the model's ability to predict credit card approval. We will grid search over the following two:

• tol

• max_iter

We will instruct GridSearchCV() to perform a [cross-validation](https://www.dataschool.io/machine-learning-with-scikit-learn/) of five folds, to determine the best possible result of classification accuracy.
