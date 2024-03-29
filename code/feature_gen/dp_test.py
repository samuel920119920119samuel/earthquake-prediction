import numpy as np
import pandas as pd
import time
import os

start=time.time()

submission = pd.read_csv('raw_data/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(index=submission.index, dtype=np.float32,
    columns=['ave', 'std', 'max', 'min','av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
    'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
    'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
    'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
    'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])

for seg_id in X_test.index:
    seg = pd.read_csv('raw_data/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
        
    X_test.loc[seg_id, 'av_change_abs'] = np.mean(np.diff(x))
    X_test.loc[seg_id, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()
    
    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()
    
    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()
    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()
    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()
    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()
    
    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()
    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()
    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()
    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()
    
    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()
    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()
    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()
    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()

print(time.time()-start)
#from IPython import embed
#embed()
ft_dir = 'features/'
if not os.path.exists(ft_dir):
    os.mkdir(ft_dir)

X_test.to_pickle(ft_dir + 'x_test.pkl')
