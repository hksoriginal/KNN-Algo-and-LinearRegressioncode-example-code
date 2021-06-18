import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
X = iris["data"][:,3:]
Y = (iris["target"]==2).astype(np.int64)

clf = LogisticRegression()
clf.fit(X,Y)
pred = clf.predict([[2.25]])


print("Logistic Regression Predicted:")
if (pred == [0]):
    print("\tNot Iris-Verginica")
elif(pred==[1]):
    print("\tIris-Verginica")

                