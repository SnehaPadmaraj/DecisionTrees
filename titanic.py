import os
os.getcwd()
os.chdir("C:/Users/sneha/Desktop/machine learning lab")
os.getcwd()
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix
data1 = pd.read_csv("titanic.csv")
data1 = data1.drop(["PassengerId", "Pclass", "Name", "SibSp", "Parch","Ticket","Fare", "Cabin"], axis = 1)
print(data1.isnull().sum())
data1.fillna(data1.median(),inplace = True)
y = data1[["Survived"]]
x = data1.drop(["Survived"], axis = 1)
x = pd.get_dummies(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
lm = LogisticRegression()
lm.fit(xtrain,ytrain)
prediction_value = lm.predict(xtest)
print(confusion_matrix(ytest, prediction_value))
print(accuracy_score(ytest, prediction_value))
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
tree=DecisionTreeClassifier()
tree.fit(xtrain,xtrain)
prediction=tree.predict(xtest)
print('The accuracy of the Decision Tree is',accuracy_score(prediction_value,ytest))
svc=SVC()
svc.fit(xtrain,ytrain) 
prediction=svc.predict(xtest)
print('The accuracy of the SVC is',accuracy_score(prediction_value,ytest))
forest=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
forest.fit(xtrain,xtrain)
print('The accuracy of the SVC is',accuracy_score(prediction,ytest))

