import numpy as np
import pandas as pd
import time
import gc

start_time=time.time()
gc.disable()#speed up by disable gc temporarily
train = pd.read_csv('raw_data/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})#fp32 is enough
gc.enable()
print(time.time()-start_time)

rows = 150_000#150,000
samples = 10000000
segments = int(np.floor(train.shape[0] / rows))


X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
    columns=['ave', 'std', 'max', 'min', 'av_change_rate',
    'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000'])
Y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,columns=['time_to_failure'])

for start,end in zip(range(samples),range(rows,samples+rows)):
    seg = train.iloc[start:end]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y_tr.loc[segment, 'time_to_failure'] = y
    X_tr.loc[segment, 'ave'] = x.mean()
    X_tr.loc[segment, 'std'] = x.std()
    X_tr.loc[segment, 'max'] = x.max()
    X_tr.loc[segment, 'min'] = x.min()
    
    X_tr.loc[segment, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    
    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
    if(start%150000==0):
        print(start)
        print(time.time()-start_time)

print(time.time()-start)
#from IPython import embed
#embed()
X_tr.to_pickle('x_tr.pkl')
Y_tr.to_pickle('y_tr.pkl')