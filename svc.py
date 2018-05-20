import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

features = pd.read_csv('data/nonlinsep-traindata.csv').values
classes = pd.read_csv('data/nonlinsep-trainclass.csv').values

X_train, X_test, y_train, y_test = train_test_split(features, classes)

svc = SVC(kernel='poly')
svc.fit(X_train, y_train.ravel())

# Evaluation metrics
print("Confusion matrix: ")
print(confusion_matrix(y_test.ravel(), svc.predict(X_test)))
print(classification_report(y_test.ravel(), svc.predict(X_test)))


plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVC classifier - test data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=svc.predict(X_test))
plt.show()


plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVC classifier - all data')
plt.scatter(features[:, 0], features[:, 1], c=svc.predict(features))
plt.show()
