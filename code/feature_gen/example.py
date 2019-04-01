import numpy as np
import pandas as pd

X_tr=pd.read_pickle('x_tr.pkl')
X_test=pd.read_pickle('x_test.pkl')
Y_tr=pd.read_pickle('y_tr.pkl')

print(X_tr.iloc[0])
print(X_test.loc[X_test.iloc[0].name])
print(Y_tr.head())

from IPython import embed
embed()