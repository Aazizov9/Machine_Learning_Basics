import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


x = pd.read_csv("features_train.csv").iloc[:, 2:1002]
x = np.array(x)


y = pd.read_csv("features_train.csv").iloc[:, -1]
y = np.array(y)


test_x = pd.read_csv("features_test.csv").iloc[:, 2:1002]
test_x = np.array(test_x)

test_y = pd.read_csv("features_test.csv").iloc[:, -1]
test_y = np.array(test_y)

SVC_model = SVC()
SVC_model.fit(x, y)

SVC_prediction = SVC_model.predict(test_x)
print("SVC "+ str (accuracy_score(SVC_prediction, test_y))) 




KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x, y)
KNN_prediction = KNN_model.predict(test_x)
print("KNN: " + str (accuracy_score(KNN_prediction, test_y)))



from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

clf = RandomForestClassifier()

voting_clf = VotingClassifier(estimators=[('SVC', SVC_model), ('KNN_model', KNN_model), ('clf', clf)])
voting_clf.fit(x, y)
preds = voting_clf.predict(test_x)

print("voting_clf: " + str (accuracy_score(preds, test_y)))