# load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
 
X = iris.data
y = iris.target
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
 
y_pred = gnb.predict(X_test)
 
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy


# Gaussian Naive Bayes model accuracy(in %): 95.0
