import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
df = pd.read_csv('passengers.csv')

# Update sex column to numerical
df.Sex.replace('male', 0, inplace=True)
df.Sex.replace('female', 1, inplace=True)

# Fill the nan values in the age column
df.Age.fillna(df.Age.mean(), inplace=True)

# Create a first/second class column
df['FirstClass'] = df.Pclass.apply(lambda x: 1 if x == 1 else 0)
df['SecondClass'] = df.Pclass.apply(lambda x: 1 if x == 2 else 0)
# Select the desired features
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = df['Survived']

# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size = 0.2, random_state=50)

# Scale the feature data so it has mean = 0 and standard deviation = 1
norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Score the model 
model.score(X_train, y_train)
model.score(X_test, y_test)

# Analyse coefficients
list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0]))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose])

# Scale the sample passenger features
sample_passengers = norm.transform(sample_passengers)

# Make survival predictions!
prediction = model.predict(sample_passengers)
prediction_prob = model.predict_proba(sample_passengers)
print(prediction)
print(prediction_prob)