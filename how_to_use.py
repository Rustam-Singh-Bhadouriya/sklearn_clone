from linear_model.LinearRegression import SimpleLinearRegression # Importing Regression
from preprocessing.Scaler import MinMaxScaler # Importing Scaler
import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([10 , 20, 30, 40, 50, 60])
y = np.array([100, 200, 300 ,400 ,500, 600])
X_test = np.array([60, 70, 80])

# Without Scaler - Not Prefered

Model = SimpleLinearRegression()
Model_sk = LinearRegression()

Model.fit(X, y)
print(Model.get_weight_bias())
print(Model.predict(X_test))

Model_sk.fit(X.reshape(-1, 1), y.reshape(-1, 1))
print(Model_sk.coef_, Model_sk.intercept_)
print(Model_sk.predict(X_test.reshape(-1, 1)))

# ___________________________________________
# With Scaler - Prefered

Scaler = MinMaxScaler()
X = Scaler.fit_transform(X.reshape(-1, 1))
X_test = Scaler.transform(X_test.reshape(-1, 1))

Model.fit(X.reshape(-1), y.reshape(-1))
print(Model.get_weight_bias())
print(Model.predict(X_test))