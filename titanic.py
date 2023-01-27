import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits import mplot3d

from compare.sample import get_sample
from compare.data import load_clean_data
from compare.fit import fit_model
from compare.vis import *
from compare.k import get_best_k

## 1: LOAD, ORGANISE AND CLEAN DATA
# Load cleaned data into a dataframe
df = load_clean_data()
# TODO: Add an input for 'MASTER' than just first class. Learn what this means, does it depend on age? Class?

# Sort data into desired features and labels 
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']]
labels = df['Survived']

# Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)
 
# Normalise the feature data (mean = 0, std = 1)
norm = StandardScaler()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)


## 2: CHOOSE MODEL
# Introduction message
print("\n" + "#"*75)
print("\nYou were aboard the Titanic when it struck an iceberg! This machine learning algorithm will determine if you will survive.")
print("\nPlease pick which machine learning model you would like to use to determine your chances of survival. \n\n\n~~~CHOICES~~~\n")

# List of ML models is printed to the user
models = {'k':'K-Nearest Neighbours', 'log':'Logistic Regression', 'svm':'Support Vector Machine', 'tree':'Decision Tree'}
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


## 3: CREATE AND USE MODEL 
if model_name == 'log': 
    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for label of new (normalised) test data
    sample_features = get_sample(norm, True)
    prediction_prob = classifier.predict_proba(sample_features)
    prediction_prob = round(100*prediction_prob[0][1], 2)
    print("\nYour probability of survivial is {0}%! The accuracy of this model is {1}% and the f1 score is {2}%\n".format(prediction_prob, accuracy, f1))

elif model_name == 'svm': 
    # Fit data to model and determine accuracy 
    classifier = SVC(gamma = 0.05, C = 1000)
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label for new test data
    sample_features = get_sample(norm, True)
    if classifier.predict(sample_features) == 1:
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

elif model_name == 'k': 
    # Determine best value of k
    k = get_best_k(X_train_norm, y_train,  X_test_norm, y_test)
    print("~~~K = {} ~~~".format(k))

    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label for new test data
    sample_features = get_sample(norm, True)
    if classifier.predict(sample_features) == 1: 
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

elif model_name == 'tree': 
    classifier = LogisticRegression()

else: print("Error loading model")

