import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


## CLEAN AND SHOW DATA

# Load data into dataframe
df = pd.read_csv("honeyproduction.csv")

# See data types
# df.info()

# See if any typos
df.year.unique()

# Group year by total production mean
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Create X and y variables. X must be a 2D array for LR model
y = prod_per_year.totalprod 
X = prod_per_year.year
X = X.values.reshape(-1, 1)

# Plot graph
plt.scatter(X, y)
plt.title("Honey production per year")
plt.xlabel("Year")
plt.ylabel("Total Production")


## FIT A LINEAR REGRESSION MODEL

model = linear_model.LinearRegression()
fit = model.fit(X, y)
y_predict = fit.predict(X)
plt.plot(X, y_predict, c='r')
plt.show()

## PREDICT

print(fit.predict(2050))
#


