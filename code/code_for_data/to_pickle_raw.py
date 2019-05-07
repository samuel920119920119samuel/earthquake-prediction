import pickle
import time
from itertools import islice

path='train.csv'

x=list()
y=list()

start_time=time.time()

with open(path) as f:
    for line in islice(f,1,629145481):
        #print(line.strip())
        a,b=line.strip().split(',')
        if(len(x)%1000==999):
            y.append(float(b))
        x.append(int(a))
        #y.append(float(b))
        if(len(x)%1000000==0):
            print(str(len(x))+':'+str(time.time()-start_time))

while(len(x)%1000!=0):
    x.pop(-1)
    #y.pop(-1)
with open('train_data_raw.pickle', 'wb') as file_pi:
    pickle.dump((x,y),file_pi)