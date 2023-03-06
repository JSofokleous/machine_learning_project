import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## DATA
streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")
df = pd.DataFrame(streeteasy)
print(df.columns.values)
print(df.head())
print(len(df))
x = df[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'has_roofdeck', 'has_doorman', 'has_patio', 'has_gym' ]]
y = df[['rent']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)


## LINEAR REGRESSION
lm = LinearRegression()
model = lm.fit(x_train, y_train)
y_predict= lm.predict(x_test)
print("Train score: ", lm.score(x_train, y_train))
print("Test score: ", lm.score(x_test, y_test))


## PLOT
plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")
plt.show()