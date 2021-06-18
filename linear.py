import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabetes = datasets.load_diabetes()
# print(diabetes.DESCR)p

diabetes_x = diabetes.data

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted =  model.predict(diabetes_x_test)

print("Means Squared Error is : ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# plt.scatter(diabetes_x_test,diabetes_y_test)
# plt.plot(diabetes_x_test,diabetes_y_predicted)
# plt.show()


# Means Squared Error is :  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698