import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from compare.data import load_clean_data
from compare.fit import fit_model

## 1: LOAD, ORGANISE AND CLEAN DATA
# Load cleaned data into a dataframe
df = load_clean_data()

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
models = {'log':'Logistic Regression'}
for i in models:
    print("For a {} model, please write \"{}\"".format(models[i], i))
print("\n~~~~~~~~~~~~~")

# User prompted to choose a ML model, which is only accepted if a preset string
while True:
    model_name = input("\nEnter here: ")
    if model_name in models: 
        print("You have picked a {} model!".format(models[i]))
        break
    print("Please pick a valid model")

# Create model user selected
if model_name == 'log': model = LogisticRegression()
elif model_name == 'k': model = LogisticRegression()
elif model_name == 'svm': model = LogisticRegression()
elif model_name == 'tree': model = LogisticRegression()


##3: FIT DATA TO MODEL AND DETERMINE ACCURACY
accuracy, f1 = fit_model(model, X_train, y_train, X_test, y_test)


## 4: PREDICT FOR NEW TEST DATA
# Example passenger features (not used later on)
Jack = np.array([0, 20, 0, 0, 0])
Rose = np.array([1, 17, 1, 0, 0])
example_passengers = np.array([Jack, Rose])
example_passengers = norm.transform(example_passengers)

# Take input for a character with features: Name, age, sex and class
sample_name = input("\nWhat is your character's name? ")
sample_age = 0
while sample_age <= 0:
    sample_age = int(input("\nWhat is the age of your character? "))
sample_sex = -1
while sample_sex < 0 or sample_sex > 1:
    sample_sex = float(input("\nWhat is the sex of your character? (Please enter 0 if male, 1 if female, or any number in between if non-binary): "))
sample_class = 0
while sample_class != 1 and sample_class != 2 and sample_class != 3:
    sample_class = float(input("\nWhat is the class of your character? (Please enter 1 for first, 2 for second, or 3 for third): "))

# Determine class from input
sample_first_class = 0
sample_second_class = 0
if sample_class == 1: sample_first_class = 1
elif sample_class == 2: sample_second_class = 1

# Create and normalise 2D array for sample character featires
sample_passenger = np.array([[sample_sex, sample_age, sample_first_class, sample_second_class, sample_first_class]])
sample_passenger = norm.transform(sample_passenger)

# Make survival predictions. Round and convert to 1.d.p
prediction_prob = model.predict_proba(sample_passenger)
prediction_prob = round(100*prediction_prob[0][1], 2)
print("\n\nThe probability of survivial for {0} is {1}%\nThe accuracy of this model is {2}% and the f1 score is {3}%\n".format(sample_name, prediction_prob, accuracy, f1))

# TODO: Add an input for 'MASTER' than just first class. Learn what this means, does it depend on age? Class?

