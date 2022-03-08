import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

print('Loading data ...')
X, y = datasets.load_iris(return_X_y=True)
print('X.shape = ' + str(X.shape) + ' y.shape = ' + str(y.shape))

print('Split data')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print('Processing: svm.SVC')
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print('Done!')

score = clf.score(X_test, y_test)
print('score = ' + str(score))