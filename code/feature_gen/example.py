import numpy as np
import pandas as pd

ft_dir = 'features/'
X_tr   = pd.read_pickle(ft_dir + 'x_tr.pkl')
X_test = pd.read_pickle(ft_dir + 'x_test.pkl')
Y_tr   = pd.read_pickle(ft_dir + 'y_tr.pkl')

print(X_tr.iloc[0])
print(X_test.loc[X_test.iloc[0].name])
print(Y_tr.head())

# from IPython import embed
# embed()
