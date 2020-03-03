from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

X = np.linspace(0, 1, 10)
print(X)

kf = KFold(n_splits=4)

for train, test in kf.split(X):
    print("%s %s\n" % (X[train], X[test]))


