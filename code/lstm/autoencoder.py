from tensorflow.keras.layers import Input,cuDNNGRU, RepeatVector
from tensorflow.keras.models import Model

# argument
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--partial", help="Use only a part of dataset", action="store_true")
args = parser.parse_args()
if args.partial:
    print("Reading 30% of dataset...")
    if os.path.isfile('raw_data/partial_train.parquet.gzip'):
        float_data =  pd.read_parquet('raw_data/partial_train.parquet.gzip').to_numpy()
    else:
        float_data = pd.read_csv("raw_data/train.csv", \
                                 dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}, \
                                 nrows=204677355)
        float_data.to_parquet('raw_data/partial_train.parquet.gzip', compression='gzip')
        float_data = float_data.to_numpy()
else:
    print("Reading full dataset...")
    if os.path.isfile('raw_data/full_train.parquet.gzip'):
        float_data =  pd.read_parquet('raw_data/full_train.parquet.gzip').to_numpy()
    else:
        float_data = pd.read_csv("raw_data/train.csv", \
                                 dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
        float_data.to_parquet('raw_data/full_train.parquet.gzip', compression='gzip')
        float_data = float_data.to_numpy()

print('read raw data complete')



float_data = 

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
