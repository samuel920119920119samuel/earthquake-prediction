import pickle
import time
from scipy.stats import kurtosis
from collections import deque
import numpy as np

path='train_data_raw.pickle'


start_time=time.time()

def compute_features(x):
    tmp=np.array(x)
    return [tmp.mean(),tmp.max(),tmp.min(),tmp.std(),np.diff(tmp).mean(),kurtosis(x)]
    #return [0,1]
    pass

buffer=deque(maxlen=1000)
i=0
x_raw,y_raw=pickle.load(open(path,'rb'))
x=list()
y=list()
batch_x=list()
batch_y=list()
for a,b in zip(x_raw,y_raw):
    i+=1
    buffer.append(a)
    if(len(buffer)>=1000):
        tmp=np.array(buffer)
        batch_x.append([tmp.mean(),tmp.max(),tmp.min(),tmp.std(),np.diff(tmp).mean(),kurtosis(tmp)])
        batch_y.append([b])
        [buffer.popleft() for i in range(100)]
        if(len(batch_x)>=150):
            x.append(list(batch_x))
            y.append(list(batch_y))
            batch_x=list()
            batch_y=list()
    if(i%1000000==0):
        print(str(i)+':'+str(time.time()-start_time))
#    if(i%10000000==0):
#        with open('train_data_tmp_'+str(i)+'.pickle', 'wb') as file_pi:
#            pickle.dump((x,y),file_pi)

x=np.array(x,dtype='float32')
y=np.array(y,dtype='float32')
with open('train_data.pickle', 'wb') as file_pi:
    pickle.dump((x,y),file_pi)