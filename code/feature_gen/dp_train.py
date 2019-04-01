import numpy as np
import pandas as pd
import time
import gc

start=time.time()
gc.disable()#speed up by disable gc temporarily
train = pd.read_csv('raw_data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})#fp32 is enough
gc.enable()
print(time.time()-start)

rows = 150_000#150,000
segments = int(np.floor(train.shape[0] / rows))


X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
    columns=['ave', 'std', 'max', 'min','av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
    'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
    'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
    'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
    'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])
Y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()
total_std = train['acoustic_data'].std()
total_max = train['acoustic_data'].max()
total_min = train['acoustic_data'].min()
total_sum = train['acoustic_data'].sum()
total_abs_max = np.abs(train['acoustic_data']).sum()

for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y_tr.loc[segment, 'time_to_failure'] = y
    X_tr.loc[segment, 'ave'] = x.mean()
    X_tr.loc[segment, 'std'] = x.std()
    X_tr.loc[segment, 'max'] = x.max()
    X_tr.loc[segment, 'min'] = x.min()
    
    
    X_tr.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))
    X_tr.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()
    X_tr.loc[segment, 'abs_min'] = np.abs(x).min()
    
    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
    
    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
    
    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()
    
    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()

print(time.time()-start)
#from IPython import embed
#embed()
X_tr.to_pickle('x_tr.pkl')
Y_tr.to_pickle('y_tr.pkl')