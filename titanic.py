import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


## 1: LOAD, ORGANISE AND CLEAN DATA
# Load the passenger data
df = pd.read_csv('passengers.csv')

# Update sex column to numerical
df.Sex.replace('male', 0, inplace=True)
df.Sex.replace('female', 1, inplace=True)

# Fill the nan values in the age column to be mean age
df.Age.fillna(df.Age.mean(), inplace=True)

# Create a first/second class column
df['FirstClass'] = df.Pclass.apply(lambda x: 1 if x == 1 else 0)
df['SecondClass'] = df.Pclass.apply(lambda x: 1 if x == 2 else 0)

# Create a surname column (not applicable to LR model)
df['Surname'] = df.Name.apply(lambda x: x.split()[0].strip(','))

# Create a master column
df['Master'] = df.Name.apply(lambda x: 1 if x.split()[1].strip('.') == 'Master' else 0)
    # Increases accuracy from 87.8% to 88.5% 
    # Increases F1 score from 81.6% to 83.1%


## 2: FIT A LOGISTIC REGRESSION MODEL
# Select the desired features
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'Master']]
survival = df['Survived']

# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size = 0.2, random_state=50)

# Scale the feature data (so it has mean = 0 and standard deviation = 1)
norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Score the model 
model.score(X_train, y_train)
model.score(X_test, y_test)

# Analyse coefficients: Sex and First class have the strongest correlation
list(zip(['Sex','Age','FirstClass','SecondClass', 'Master'],model.coef_[0]))


## 3: DETERMINE ACCURACY OF MODEL
# Predict labels using test data
y_pred = model.predict(X_test)

# Determine accuracy and F1 score, Round to 1.d.p and convert to percentage 
accuracy = accuracy_score(y_test, y_pred)
accuracy = round(100*accuracy, 1)
f1 = f1_score(y_test, y_pred)
f1 = round(100*f1, 1)



## 4: PREDICT FOR SAMPLE PASSANGERS
# Example passenger features (not used later on)
Jack = np.array([0, 20, 0, 0, 0])
Rose = np.array([1, 17, 1, 0, 0])
example_passengers = np.array([Jack, Rose])
example_passengers = norm.transform(example_passengers)

# Take input for a character with features: Name, age, sex and class
print("\nYou were aboard the Titanic when it struck an iceberg! This machine learning algorithm will determine if you will survive.")
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
print("\nThe survivial rate for {0} is: {1}%.\nThe accuracy of this model is {2}% and the f1 score is {3}%".format(sample_name, prediction_prob, accuracy, f1))