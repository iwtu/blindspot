# Adrian Lachata
#
# inspiration:
# https://jtsulliv.github.io/perceptron/
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix


features_path = 'data/linsep-traindata.csv'
classes_path = 'data/linsep-trainclass.csv'

features = pd.read_csv(features_path).values
classes = pd.read_csv(classes_path).values

X_train, X_test, y_train, y_test = train_test_split(features, classes)

perceptron = Perceptron(max_iter=100, tol=None)
perceptron.fit(X_train, y_train.ravel())

# Evaluation metrics
print("Confusion matrix: ")
print(confusion_matrix(y_test.ravel(), perceptron.predict(X_test)))
print(classification_report(y_test.ravel(), perceptron.predict(X_test)))

# chart depicting the classifier as well as the samples in 2D
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron classifier')
plt.scatter(features[:, 0], features[:, 1], c=perceptron.predict(features))
plt.show()

