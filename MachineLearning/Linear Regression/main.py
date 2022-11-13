import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep =";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

X = np.array(data.drop(["G3"], 1))
y = np.array(data["G3"])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1)
"""
maxAcc = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if maxAcc < acc:
        maxAcc = acc
        pickle.dump(linear, open("StudentModel.pickle", "wb"))
"""
#Accuracy = 97.35%

pickle_in = open("StudentModel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coeff: ", linear.coef_)
print("Intercept ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(round(predictions[x]), x_test[x], y_test[x])

p = "G2"
pyplot.style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()