import numpy as np
from tqdm import tqdm

# n_step step_len

def extract_feature(z):
    return np.c_[z.mean(axis=1), 
                 z.min(axis=1),
                 z.max(axis=1),
                 z.std(axis=1)]

def create_X(x, last_index=None, n_steps=300, step_len=500):
    if last_index == None:
        last_index = len(x)
    start_index = last_index - n_steps * step_len
    assert start_index >= 0

    tmp = (x[start_index:last_index].reshape(n_steps, -1))
    return np.c_[extract_feature(tmp),
                 extract_feature(tmp[:, -step_len//5:]),
                 extract_feature(tmp[:, -step_len//10:]),
                 extract_feature(tmp[:, -step_len//50:]),
                 extract_feature(tmp[:, -step_len//100:])]

def gen(data, min_index=0, max_index=None, batch_size=32, n_steps=300, step_len=500, n_features=20):
    if max_index is None:
        max_index = len(data) - 1
    while True:
        rows = []
        while len(rows) < batch_size:
            candidate = np.random.randint(min_index + n_steps * step_len, max_index)
            rows.append(candidate)
        rows = np.array(rows, dtype=np.int)
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        for i, row in enumerate(rows):
            samples[i] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_len=step_len)
            targets[i] = data[row-1, 1]
        yield samples, targets 
