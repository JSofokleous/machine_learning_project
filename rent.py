import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1
from compare.load_data import load_titanic, load_house
# 3
from compare.input_sample import get_binary_sample
# 4
from compare.fit_model import fit_score_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from compare.k import *
from sklearn.svm import SVC
from compare.vis import *



## 1: LOAD DATA
# Load cleaned data into a dataframe
features, labels = load_house()

# Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)

# Normalise the feature data (mean = 0, std = 1 using Z-score method), only fit StandardScalar to train data.
normalise = StandardScaler()
X_train_norm = normalise.fit_transform(X_train)
X_test_norm = normalise.transform(X_test)

# List current selected features
feature_names = [col for col in features.columns]
print("Feature names: ", feature_names)


## 2: CHOOSE MODEL
# Introduction message: List of ML models is printed to the user
print("\n" + "#"*75)
print("\nYou were aboard the Titanic when it struck an iceberg! This machine learning algorithm will determine if you will survive.")
print("\nPlease pick which machine learning model you would like to use to determine your chances of survival. \n\n\n~~~CHOICES~~~\n")
models = {'knn':'K-Nearest Neighbours', 'log':'Logistic Regression', 'svm':'Support Vector Machine', 'tree':'Decision Tree'}
for i in models:
    print("For the {} model, please write \"{}\"".format(models[i], i))
print("\n~~~~~~~~~~~~~")

# User prompted to choose a ML model, which is only accepted if a preset string
while True:
    model_name = input("\nEnter here: ")
    if model_name in models: 
        print("You have picked a {} model!".format(models[model_name]))
        break
    print("Please pick a valid model")


## 3: INPUT SAMPLE

# Return user input sample for selected features, and normalise
# sample_features = get_binary_sample(feature_names)     # UNCOMMENT TO SELECT CUSTOM FEATURES
sample_features = [1 for col in features.columns]     # UNCOMMENT FOR DEFAULT FEATURES: 1st class female, 1yo  
sample_features = normalise.transform([sample_features])


## 4: CREATE AND USE MODEL 

# Logistic Regression Model
if model_name == 'log': 
    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data. [[failure, survival]]
    prediction_prob = classifier.predict_proba(sample_features)
    prediction_prob = round(100*prediction_prob[0][1], 2)
    print("\nYour probability of survivial is {0}%! The accuracy of this model is {1}% and the f1 score is {2}%\n".format(prediction_prob, accuracy, f1))

# K-Nearest Neighbors Model
# PROBLEM: Only binary thus only 8 possibe combinations, 2**3. only 4 for SVM. 
elif model_name == 'knn': 
    # Determine best value of k
    k = get_best_k(X_train_norm, y_train,  X_test_norm, y_test)
    print("~~~K = {}~~~".format(k))

    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample using own KNN model
    if classify(sample_features, X_train_norm, y_train, k) == 1: 
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

# Support Vector Model
elif model_name == 'svm': 
    # Only allow 2 features to compare (automatically set to first 2 columns)
    print("Picking first 2 features: {0} and {1}".format(feature_names[:2][0], feature_names[:2][1]))
    X_train_norm = np.vstack([X_train_norm[:,0], X_train_norm[:,1]]).T
    X_test_norm = np.vstack([X_test_norm[:,0], X_test_norm[:,1]]).T
    sample_features = [sample_features[0][:2]]

    # Fit data to model and determine accuracy 
    classifier = SVC(kernel='linear', C = 0.01)
    # classifier = SVC(kernel='rbf', gamma = 0.05, C = 1000)
    accuracy, f1 = fit_score_model(classifier, X_train_norm, y_train, X_test_norm, y_test)
    make_three(classifier, X_train_norm, y_train)
    make_two(classifier, X_train_norm, y_train)

    # Predict label of sample
    if classifier.predict(sample_features) == 1:
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

# Tree
elif model_name == 'tree': 
    print("This feature is coming soon!")

# Error
else: print("Error loading model")