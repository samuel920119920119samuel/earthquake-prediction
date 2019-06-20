import numpy as np
import pandas as pd
import time

start=time.time()

submission = pd.read_csv('raw_data/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(index=submission.index, dtype=np.float32,
    columns=['ave', 'std', 'max', 'min', 'av_change_rate',
    'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000'])

for seg_id in X_test.index:
    seg = pd.read_csv('raw_data/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
        
    X_test.loc[seg_id, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    
    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()

print(time.time()-start)
#from IPython import embed
#embed()
X_test.to_pickle('x_test.pkl')