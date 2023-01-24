import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Import data
streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")
df = pd.DataFrame(streeteasy)
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = df[['rent']]

# Split data into 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, random_state=6)

# Fit linear regression to test data
mlr = LinearRegression()
mlr.fit(x_train, y_train) 

# Test for one case
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
sonny_predict = mlr.predict(sonny_apartment)
print("Predicted rent: $%.2f" % sonny_predict)

# Test model on test data
y_predict = mlr.predict(x_test)
# plt.scatter(y_test, y_predict)
# plt.xlabel("Prices: $Y_i$")
# plt.ylabel("Predicted prices: $\hat{Y}_i$")
# plt.title("Actual Rent vs Predicted Rent")
# plt.show()

# Viewing how different affect rent
plt.scatter(df[['has_elevator']], df[['rent']], c='r')
plt.ylabel("Rent")
plt.show()

