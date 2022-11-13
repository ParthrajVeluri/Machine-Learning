import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

#Transforms string data into numerical data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#zip creates an array filled with tuples such that it would contain from, zip(a,b,...c)
#the (a,b,...c) as the tuple elements

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
print(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
predicted = model.predict(X_test)
print(acc)

for x in range(len(X_test)):
    print("Predicted ", predicted[x], "Data: ", X_test[x], "Actual: ", y_test[x])



