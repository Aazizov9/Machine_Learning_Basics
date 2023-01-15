import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from scipy.stats import uniform

from sklearn.model_selection import train_test_split as split

x = pd.read_csv("train.csv").drop(["id", "label"], axis=1)
x = np.array(x.fillna(x.mean()))

y = pd.read_csv("train.csv")
y = np.array(y['label'].fillna(y.mean()))

test = pd.read_csv("test.csv").drop(["id"], axis=1)


SVC_model = SVC()
SVC_model.fit(x, y)

SVC_prediction = SVC_model.predict(test)
ID = range(len(SVC_prediction))
Ypd = pd.DataFrame({'id': ID})
Ypd['label'] = SVC_prediction
Ypd.to_csv('Azizov_Amir.csv', index=False)