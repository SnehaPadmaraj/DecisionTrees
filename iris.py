from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
d1 = load_iris()
y = d1.target
y
x = pd.DataFrame(d1.data,columns=d1.feature_names)
x
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20, random_state = 0)
xtrain.shape
xtest.shape
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 0)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred))
from sklearn.ensemble import RandomForestClassifier 
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(xtrain, ytrain)
