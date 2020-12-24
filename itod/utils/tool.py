"""Useful tools includes 'stat_data', 'print', etc.
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import csv
import os
import pickle
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from functools import wraps
import resource
import numpy as np
import pandas as pd
import pytz


def stat_data(data=None, name='data'):
    #
    # import inspect
    # a= inspect.signature(stat_data)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 100)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # without scientific notation

    columns = ['col_' + str(i) for i in range(data.shape[1])]
    dataset = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
    print(f'{name}.shape: {data.shape}')
    print(dataset.describe())
    print(dataset.info(verbose=True))


def progress_bar(i, num=100):
    """Print progress bar

    Parameters
    ----------
    i:
        current progress
    num:
        all num
    """
    if num <= 0:
        msg = f'num: {num} <=0'
        raise ZeroDivisionError(msg)
    percent = (i) / num
    # print(f'{i + 1}/{num}, percent: {percent * 100:.2f}%')
    status_bar = '-' * 40
    per = int(percent * 40)
    per_bar = '=' * per + status_bar[per:]
    print(f'{per_bar} : {percent * 100:.2f}% = {i}/{num}')


def _pprint(params, offset=0, printer=repr, sorted_flg=False):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print

    offset : int
        The offset in characters to add at the begin of each line.

    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    if sorted_flg:
        items = sorted(params.items())
    else:
        items = params.items()
    for i, (k, v) in enumerate(items):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        else:  # i == 0
            params_list.append(' ')
            this_line_length += 2

        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


def pprint(params_dict=OrderedDict(), name='func_name', type='func', sorted_flg=False, verbose=True):
    if type == 'func':
        print(f'{name}\'s parameters: ')
    else:
        print(f'{name}: ')
    params_str = _pprint(params_dict, sorted_flg=sorted_flg)
    print(f'{params_str}')


def pprint2(*args, name='', sep=' ', end='\n', file=sys.stdout, flush=True):
    if len(name) > 0:
        print(f'{name}: {args}', sep=sep, end=end, file=file, flush=flush)
    else:
        print(args, sep=sep, end=end, file=file, flush=flush)


def check_n_generate_path(file_path, overwrite=True):
    """Check if a path is existed or not, and generate all directories in the path

    Parameters
    ----------
    file_path
    overwrite

    Returns
    -------

    """
    path_dir = os.path.dirname(file_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if os.path.exists(file_path):
        if overwrite:
            os.remove(file_path)

    return file_path


def del_obj(obj):
    # if obj in globals():
    #     del obj
    try:
        if len(obj) > 0:
            del obj
    except Exception as e:
        print(f'del {obj} failed: {e}.')


def execute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        st = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
        print(f'\'{func.__name__}()\' starts at {st}')
        result = func(*args, **kwargs)
        end = time.time()
        ed = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
        tot_time = (end - start) / 60
        tot_time = float(f'{tot_time:.4f}')
        print(f'\'{func.__name__}()\' ends at {ed}. It takes {tot_time} mins')
        return result

    return wrapper


def func_notation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'>>> enter \'{func.__name__}() in {os.path.relpath(func.__code__.co_filename)} '
              f'at line {func.__code__.co_firstlineno}\'')
        result = func(*args, **kwargs)
        print(f'<<< exit \'{func.__name__}() in {os.path.relpath(func.__code__.co_filename)} '
              f'at line {func.__code__.co_firstlineno}\'')
        return result

    return wrapper


def memory_usuage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'---memory analysis: \'{func.__name__}() in {os.path.relpath(func.__code__.co_filename)}\'')
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        result = func(*args, **kwargs)
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        # print(f'--- exit \'{func.__name__}() in {os.path.relpath(func.__code__.co_filename)} '
        #       f'costs maxrss: {(r1.ru_maxrss-r0.ru_maxrss)/(1024**1):.2f}MB.\'')
        keys = ['ru_utime', 'u_stime', 'ru_maxrss', 'ru_ixrss', 'ru_idrss', 'ru_isrss', 'ru_minflt',
                'ru_majflt', 'ru_nswap', 'ru_inblock', 'ru_unblock', 'ru_msgsnd', 'ru_msgrcv', 'ru_nsignals',
                'ru_nvcsw', 'ru_nivcsw']

        values = OrderedDict()
        for v0, v1, k in zip(r0, r1, keys):
            if k == 'ru_maxrss':
                # v = f'{(v1-v0)/(1024**2):.2f}MB'
                v = f'{v1 / (1024 ** 2):.2f}MB'
            else:
                v = f'{v1 - v0:.2f}'
            values[k] = v
        print(f'--- exit \'{func.__name__}() in {os.path.relpath(func.__code__.co_filename)} '
              f', it costs: {values}\'')
        return result

    return wrapper


def check_value():
    pass


def remove_redundant_flows(input_file='labels_csv', output_file='', verbose=True):
    """ only keep the given srcIP or srcIPs, and reduce file size

    :param input_file: labels_csv
    :param output_file: five_tuple + labels
    :return:
    """
    if verbose:
        funcparams_dict = {'input_file': input_file, 'output_file': output_file,
                           'verbose': verbose}
        pprint(OrderedDict(funcparams_dict), name=remove_redundant_flows.__name__)

    assert os.path.exists(input_file)
    assert output_file == '', f'output_file is {output_file}, please check and retry'
    check_n_generate_path(file_path=output_file, overwrite=True)

    df = pd.read_csv(input_file)
    cols_name = list(df.columns)  # 'Source IP'
    five_tuple_labels = [cols_name[2], cols_name[3], cols_name[4], cols_name[5], cols_name[6],
                         cols_name[-1]]  # five_tuple, labels
    print(f'five_tuple_labels:{five_tuple_labels}')

    df = df[five_tuple_labels]
    print(f'before removing redundant five tuples: {len(df)}')
    # dropping duplicate values
    df.drop_duplicates(keep=False, inplace=True)
    print(f'after removing redundant five tuples: {len(df)}')

    df.to_csv(output_file)

    return output_file


def transform_params_to_str(data_dict,
                            remove_lst=['verbose', 'random_state', 'means_init', None, 'auto', 'quantile'],
                            param_limit=50):
    # param_limit = 50  # # limit the length of the params string, which is used for file name

    params_dict = deepcopy(data_dict)
    params_str = ''
    for i, (key, value) in enumerate(params_dict.items()):
        if key not in remove_lst and value not in remove_lst:
            if type(value) == float or type(value) == np.float64:
                value = f'{value:.4f}'
            params_str += f'{key}:{value}, '
    if len(params_str) > param_limit:  # limit the length of the output_file name
        params_str = params_str[:param_limit]

    return params_str


def rm_outdated_files(dir_path, dur=7 * (24 * 60 * 60)):
    def convert_bytes(num):
        """
        this function will convert bytes to MB.... GB... etc
        """
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x)
            num /= 1024.0

    try:
        i = 0
        file_sizes = OrderedDict()
        for (subdir_path, dirnames, filenames) in os.walk(dir_path):
            for file_name in [os.path.join(subdir_path, file) for file in filenames]:
                if os.stat(file_name).st_mtime < time.time() - dur:
                    if '.dat' in file_name or '.pdf' in file_name or '.pth' in file_name:  # don't remove label.csv and xxx.pcap
                        f_size = os.path.getsize(file_name)  # return file size in bytes
                        if file_name not in file_sizes.keys():
                            file_sizes[file_name] = f_size
                        os.remove(file_name)
                        i += 1
                        print(
                            f'remove {file_name} (its size is {convert_bytes(f_size)}), '
                            f'because its time is more than {dur}s, ({int(dur / (24 * 3600))} days)')

        total_size = sum([v for v in file_sizes.values()])
        print(f'remove {i} old files in {dir_path}, total size is {convert_bytes(total_size)}')
        top_files = sorted(file_sizes.items(), key=lambda v: v[1], reverse=True)[:5]
        for i, (f_name, f_size) in enumerate(top_files):
            print(f'Top {i + 1} is {f_name}, {convert_bytes(f_size)} in size.')

    except Exception as e:
        print(f'Error: {e}')


@func_notation
def copy_dataset(params_i):
    """copy_datasets, avoiding access issues by multi-programs at the same time.
    params_i is a dict, so it will be updated in this function.
    """
    # new_ipt_dir = params_i['ipt_dir'] + '_' + params_i['detector_name']
    new_ipt_dir = params_i['ipt_dir']
    # if not os.path.exists(new_ipt_dir):
    #     print(f'{new_ipt_dir} doesn\'t exist.')
    #     ipt_dir_i = params_i['ipt_dir']
    #     cmd = f"cp -rf {ipt_dir_i} {new_ipt_dir}"
    #     print(f'{cmd}')
    #     try:
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    #     except Exception as e:
    #         print(f'{e}')
    #         return -1
    # else:
    #     print(f'{new_ipt_dir} exists.')
    # params_i['ipt_dir'] = new_ipt_dir
    #
    # params_i['opt_dir'] = params_i['opt_dir'] + '_' + params_i['detector_name']

    params_i['ipt_dir'] = new_ipt_dir
    params_i['opt_dir'] = params_i['opt_dir']
    print('ipt_dir:', params_i['ipt_dir'], ', opt_dir:', params_i['opt_dir'])

    return params_i


@func_notation
def get_file_path(**kwargs):
    path_dir = ''
    for (k, v) in kwargs.items():
        if k == 'file_name':
            file_name = v
            continue
        path_dir = os.path.join(path_dir, v)

    return os.path.join(path_dir, file_name)


# def get_file_path(ipt_dir='', **kwargs):
#     file_path = ipt_dir
#     for i, (key, value) in enumerate(kwargs.items()):
#         file_path = os.path.join(file_path, value)
#
#     return file_path

def get_sf_key(sf_dict, v='sf:True', header='header:False', data_cat='AGMT', dataset_name=""):
    for key, value in sf_dict.items():
        # print(key, value)
        try:
            v = value[header][data_cat][dataset_name]
            return key
        except Exception as e:
            # print("key error: ", e)
            continue

    return -1


def dump_data(data, output_file='', verbose=True):
    """Save data to file

    Parameters
    ----------
    data
    output_file
    verbose

    Returns
    -------
    output_file: path
        the file stored the data
    """
    if verbose:
        funcparams_dict = {'len(data)': len(data), 'output_file': output_file}
        pprint(funcparams_dict, name=dump_data.__name__)

    check_n_generate_path(output_file, overwrite=True)

    # # save results
    with open(output_file, 'wb') as out_hdl:
        pickle.dump(data, out_hdl)

    return output_file


@func_notation
def set_key_dict(keys, v, results):
    if len(keys) == 1:
        key = keys.pop(0)
        results[key] = v
        return results
    else:
        key = keys.pop(0)
        if key not in results.keys():
            results[key] = {}
        set_key_dict(keys, v, results[key])


@func_notation
def load_result_data(input_file):
    """ Especially for loading multi-objects stored in a file.

       :param input_file:
       :return:
       """
    results = {}
    with open(input_file, 'r') as in_hdl:
        line = in_hdl.readline()  # header
        # line = in_hdl.readline()
        i = 2
        while line:
            print(i, line.strip())
            if line == '\n':
                line = in_hdl.readline()
                i += 1
                continue
            # arr = line.split(',[')
            # keys = arr[0].split(',')
            # v = [arr[1].strip()]
            arr = line.split('] [')  # arr = line.split('] [')
            keys = arr[0][1:].split(',')
            keys = [eval(v.strip()) for v in keys]
            v = [eval(arr[1].strip()[:-1])]
            set_key_dict(keys, v, results)
            line = in_hdl.readline()
            i += 1
    return results


@func_notation
def load_txt_data(input_file):
    with open(input_file, 'rb') as in_hdl:
        while True:
            try:
                results = pickle.load(in_hdl)
            except EOFError:
                break

    return results


def set_dict(result_dict, keys, value):
    """ Save value into result_dict with keys_lst

    Parameters
    ----------
    result_dict
    keys: list
        [detector, gs, sf, ...]
    value: list

    Returns
    -------

    """
    for key in keys:
        if key in result_dict.keys():
            keys.pop(0)  # pop up the first item of the keys_lst
            set_dict(result_dict[key], keys, value)
        else:
            if len(keys) == 1:
                result_dict[key] = value
                return 0
            else:
                result_dict[key] = {}
                keys.pop(0)
                set_dict(result_dict[key], keys, value)


def get_dict(result_dict, keys=[]):
    for key, value in result_dict.items():
        if type(value) == type(dict):
            keys.append(key)
            get_dict(result_dict, keys)
        else:
            keys.append(key)
            return keys, value


def save_dict(result_dict, keys=[], out_hdl=''):
    for key, value in result_dict.items():
        if type(value) == dict:
            keys.append(key)
            save_dict(value, keys, out_hdl=out_hdl)
        else:
            keys.append(key)
            print(keys, value)
            out_hdl.write(','.join(keys) + f',{value}\n')
        keys.pop(-1)

    return keys


def time_string_to_seconds(time_string="2019-01-10 13:22:53.721089", time_format="%Y-%m-%d %H:%M:%S.%f"):
    """
    References: https://stackoverflow.com/questions/30468371/how-to-convert-python-timestamp-string-to-epoch

    Parameters
    ----------
    time_string:
    time_format:

    Returns
    -------

    utc_time = datetime.strptime("2009-03-08T00:27:31.807Z", "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()

    Z: the UTC timezone offset identitifier

    """
    date_time = datetime.strptime(time_string, time_format)
    sec = (date_time - datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, microsecond=0,
                                tzinfo=None)).total_seconds()
    # sec = (date_time - datetime(1970,1,1)).total_seconds()
    # print(sec)

    return sec


def seconds_to_time_string(second=1547148483.743690, time_format="%Y-%m-%d %H:%M:%S.%f"):
    """
    References: https://stackoverflow.com/questions/30468371/how-to-convert-python-timestamp-string-to-epoch

    Parameters
    ----------
    time_string:
    time_format:

    Returns
    -------

    utc_time = datetime.strptime("2009-03-08T00:27:31.807Z", "%Y-%m-%dT%H:%M:%S.%fZ")
    epoch_time = (utc_time - datetime(1970, 1, 1)).total_seconds()

    Z: the timezone offset identitifier

    """
    date_time = datetime.fromtimestamp(second).strftime(time_format)
    # date_time= datetime.utcfromtimestamp(second).replace(tzinfo=pytz.utc)
    print(date_time)

    return date_time


# import datetime
# import pytz

def convert_datetime_timezone(dt, tz1, tz2):
    tz1 = pytz.timezone(tz1)
    tz2 = pytz.timezone(tz2)

    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    dt = tz1.localize(dt)
    dt = dt.astimezone(tz2)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return dt


def change_timezone(second, timezone='utc'):
    """

    Parameters
    ----------
    second
    timezone

    Returns
    -------

    """
    date_time = datetime.fromtimestamp(second).replace(tzinfo=pytz.timezone(timezone.upper()))
    # print(date_time)
    second = time_string_to_seconds(str(date_time).split('+')[0])

    return second


def get_directory_depth(folder=''):
    depth = 0
    for i, sub_folder in os.listdir(folder):
        if os.path.isdir(sub_folder):
            depth += 1
            get_directory_depth(sub_folder)

    return depth


def execute_cmd(cmd):
    print(f'cmd: {cmd}')
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    except Exception as e:
        print(f'Error: {e}')
        return -1

    return result


def rm_dirs(label_dir, file_type='*'):
    if os.path.exists(label_dir):

        if file_type == '*':  # delete all files and directories
            cmd = f'rm -rf {label_dir}'
        elif file_type == 'csv':
            cmd = f"find {label_dir} -name \"*.{file_type}\" -type f -delete"
        print(f'cmd:{cmd}')

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        except Exception as e:
            print(f'Error: {e}')
            return -1
