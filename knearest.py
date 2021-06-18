
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# loading datasets
iris = datasets.load_iris()
# printing description and features


# print(iris.DESCR)
features = iris.data
lablels = iris.target

# training classifier
clf = KNeighborsClassifier()
clf.fit(features,lablels)
# giving new set of features
pred = clf.predict([[3,7,2,6]])
print("Knearest Neighbour Algorithm Predicted:")
if (pred == [0]):
    print("\tIris-Setosa")
elif(pred==[1]):
    print("\tIris-Versicolour")
elif(pred==[2]):
    print("\tIris-Virginica")
                

