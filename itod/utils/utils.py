import pickle
import pandas as pd

global pth_log, ld, lg, lw, le, lc


def dump_data(data, file_out='data.dat'):
    with open(file_out, 'wb') as f:
        pickle.dump(data, f)


def unpickle_data(file_in='data.dat'):
    with open(file_in, 'rb') as f:
        data = pickle.load(f)
        return data


def merge_dicts(dict_1, dict_2):
    for key in dict_2.keys():
        # if key not in dict_1.keys():
        dict_1[key] = dict_2[key]

    return dict_1


def data_info(data, name=''):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 100)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)  # without scientific notation

    columns = ['col_' + str(i) for i in range(data.shape[1])]
    dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
    lg(f'\n{name}.shape: {data.shape}')
    lg(f'\n{dataset.describe()}')
    lg(dataset.info(verbose=True))
