import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


## 1: LOAD DATA
# Load cleaned data into a dataframe
from compare.data import load_clean_data
df = load_clean_data()

# Sort data into desired labels and features: ['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']]
labels = df['Survived']

# Split data into train and test set (where the dataframe X holds the features, and the series y holds the labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=50)

# Normalise the feature data (Z-score method: mean = 0, std = 1), only fit StandardScalar to train data.
normalise = StandardScaler()
X_train_norm = normalise.fit_transform(X_train)
X_test_norm = normalise.transform(X_test)


## 2: CHOOSE MODEL
# Introduction message
print("\n" + "#"*75)
print("\nYou were aboard the Titanic when it struck an iceberg! This machine learning algorithm will determine if you will survive.")
print("\nPlease pick which machine learning model you would like to use to determine your chances of survival. \n\n\n~~~CHOICES~~~\n")

# List of ML models is printed to the user
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
# List current selected features
feature_names = []
for col in features.columns:
    feature_names.append(col)
print("Feature names: ", feature_names)

# Return user input sample for selected features, and normalise
from compare.sample import get_sample
sample_features = [1, 20, 1, 0, 0]
# sample_features = get_sample(feature_names)
sample_features = normalise.transform([sample_features])


## 4: CREATE AND USE MODEL 
from compare.fit import fit_model
from compare.vis import *
from compare.k import *

# Logistic Regression Model
if model_name == 'log': 
    # Fit data to model and determine accuracy 
    classifier = LogisticRegression()
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Make prediction for probability of label for sample test data
    prediction_prob = classifier.predict_proba(sample_features)
    prediction_prob = round(100*prediction_prob[0][1], 2)
    print("\nYour probability of survivial is {0}%! The accuracy of this model is {1}% and the f1 score is {2}%\n".format(prediction_prob, accuracy, f1))

elif model_name == 'svm': 
    # Only allow 2 features to compare
    X_train_norm = np.vstack([X_train_norm[:,0], X_train_norm[:,1]]).T
    X_test_norm = np.vstack([X_test_norm[:,0], X_test_norm[:,1]]).T
    sample_features = [sample_features[0][:2]]

    # Fit data to model and determine accuracy 
    classifier = SVC(kernel='linear', C = 0.01)
    # classifier = SVC(kernel='rbf', gamma = 0.05, C = 1000)
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)
    make_three(classifier, X_train_norm, y_train)
    make_two(classifier, X_train_norm, y_train)

    # Predict label of sample
    if classifier.predict(sample_features) == 1:
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

elif model_name == 'knn': 
    # Determine best value of k
    k = get_best_k(X_train_norm, y_train,  X_test_norm, y_test)
    print("~~~K = {}~~~".format(k))

    # Fit data to model and determine accuracy 
    classifier = KNeighborsClassifier(k)
    accuracy, f1 = fit_model(classifier, X_train_norm, y_train, X_test_norm, y_test)

    # Predict label of sample using own KNN model
    if classify(sample_features, X_train_norm, y_train, k) == 1: 
        print("You Survived! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))
    else: 
        print("You did not survive! The accuracy of this model is {0}% and the f1 score is {1}%\n".format(accuracy, f1))

elif model_name == 'tree': 
    print("This feature is coming soon!")

else: print("Error loading model")