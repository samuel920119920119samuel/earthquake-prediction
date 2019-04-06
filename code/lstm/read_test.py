import pandas as pd
import numpy as np
t = pd.read_csv("raw_data/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}, nrows=10).to_numpy()
print(t)
