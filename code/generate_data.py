import numpy as np
import pandas as pd

def generate_train_data():
    index = -1
    segments = 4194
    X_tr = pd.DataFrame(index = range(segments), dtype = np.float64,
                        columns = ['av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
                                  'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                                  'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                                  'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                                  'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])
    y_tr = pd.DataFrame(index = range(segments), dtype = np.float64, columns = ['time_to_failure'])
    for seg in  pd.read_csv('D://train.csv' , iterator = True , chunksize = 150000 , dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float32}):
        index = index + 1
        print(index)

        x = seg['acoustic_data'].values
        y = seg['time_to_failure'].values[-1]
        y_tr.loc[index, 'time_to_failure'] = y

        X_tr.loc[index, 'av_change_abs'] = np.mean(np.diff(x))
        X_tr.loc[index, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
        X_tr.loc[index, 'abs_max'] = np.abs(x).max()
        X_tr.loc[index, 'abs_min'] = np.abs(x).min()

        X_tr.loc[index, 'std_first_50000'] = x[:50000].std()
        X_tr.loc[index, 'std_last_50000'] = x[-50000:].std()
        X_tr.loc[index, 'std_first_10000'] = x[:10000].std()
        X_tr.loc[index, 'std_last_10000'] = x[-10000:].std()

        X_tr.loc[index, 'avg_first_50000'] = x[:50000].mean()
        X_tr.loc[index, 'avg_last_50000'] = x[-50000:].mean()
        X_tr.loc[index, 'avg_first_10000'] = x[:10000].mean()
        X_tr.loc[index, 'avg_last_10000'] = x[-10000:].mean()

        X_tr.loc[index, 'min_first_50000'] = x[:50000].min()
        X_tr.loc[index, 'min_last_50000'] = x[-50000:].min()
        X_tr.loc[index, 'min_first_10000'] = x[:10000].min()
        X_tr.loc[index, 'min_last_10000'] = x[-10000:].min()

        X_tr.loc[index, 'max_first_50000'] = x[:50000].max()
        X_tr.loc[index, 'max_last_50000'] = x[-50000:].max()
        X_tr.loc[index, 'max_first_10000'] = x[:10000].max()
        X_tr.loc[index, 'max_last_10000'] = x[-10000:].max()

    X_tr.to_csv('train_data_x.csv', index = False)
    y_tr.to_csv('train_data_y.csv', index = False)
    X_tr.to_pickle('train_data_x.pkl')
    y_tr.to_pickle('train_data_y.pkl')

def generate_test_data():
    submission = pd.read_csv('D://sample_submission.csv', index_col = 'seg_id')
    X_test = pd.DataFrame(dtype = np.float64, index = submission.index , columns = ['av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',
                                                                                    'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                                                                                    'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                                                                                    'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                                                                                    'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000'])

    for i, seg_id in enumerate(X_test.index):
        print(i)
        seg = pd.read_csv('D://test/' + seg_id + '.csv')
        x = seg['acoustic_data'].values

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

    X_test.to_csv('test_data_x.csv', index = False)
    X_test.to_pickle('test_data_x.pkl')

def main():
    generate_train_data()
    generate_test_data()

if __name__ == '__main__':
    main()





