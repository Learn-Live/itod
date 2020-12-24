import pickle


def dump_data(data, output_file='a.dat'):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
        print(data)
    return output_file


def load_data(output_file):
    with open(output_file, 'rb') as f:
        data = pickle.load(f)
        print(data)
    return data


#
output_file = '/Users/kunyang/PycharmProjects/itod/examples/data/reprst/UCHI/IOT_2019/smtv_10.42.0.1/-subflow_interval=None_q_flow_duration=0.9'
output_file = '/Users/kunyang/PycharmProjects/itod/examples/data/reprst/UCHI/IOT_2019/smtv_10.42.0.1/all-features-header:False.dat'
# # data = ([1,2], [1,3], 'a')
# # dump_data(data, out_file='a.dat')
load_data(output_file)
