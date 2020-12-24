"""Includes all configurations, such as constants and global random_state.
    1. set a random seed for os, random, and so on.
    2. print control
    3. root directory control
    4. some constants
"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

"""Step 1. random state control in order to achieve reproductive results
    ref: https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
"""
# Seed value
# Apparently you may use different seed values at each stage
random_state = 42
# 1). Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(random_state)

# 2). Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(random_state)

# 3). Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(random_state)

# 4). set torch
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""Step 2. Print control
"""
# import sys
#
# # force it to print everything in the buffer to the terminal immediately.
# sys.stdout.flush()
# # out_file = 'stdout_content.txt'
# # # sys.stdout = open(out_file, mode='w', buffering=1)
# # ###skip 'buffering' if you don't want the output_data to be flushed right away after written
# # # sys.stdout = sys.__stdout__
# #
# import functools
#
# print = functools.partial(print, flush=True)
#
# """Replace "print" with "logging"
#
# The only time that print is a better option than logging is when the goal is to display a help statement
# for a command-line application. Other reasons why logging is better than print:
#
# Ref:
#     https://docs.python.org/3/library/logging.html#logrecord-attributes
#     https://docs.python.org/3/library/logging.html#logrecord-attributes
# """
# import os
# import os.path as pth
# import sys
# import logging
# import logging.handlers as hdl
# from shutil import copy2 as cp  # use "copy2" instead of "copyfile"
#
# logger = logging.getLogger()
#
# # output log into stdout
# console_hdl = logging.StreamHandler(sys.stdout)
# console_fmt = logging.Formatter('%(asctime)s %(name)-5s [%(threadName)-10s] %(levelname)-5s %(funcName)-5s '
#                                 '%(filename)s line=%(lineno)-4d: %(message)s')
# console_hdl.setFormatter(console_fmt)
# logger.addHandler(console_hdl)
#
# # output log into file
# log_path = './log'
# log_name = pth.join(log_path, 'app.log')
# if not pth.exists(log_path):
#     os.makedirs(log_path)
# # if pth.exists(log_name):
# #     cp(log_name, log_name+'time')
# # BackupCount: if either of maxBytes or backupCount is zero, rollover never occurs
# file_hdl = hdl.RotatingFileHandler(log_name, mode='a', maxBytes=5 * 1024 * 1024,
#                                    backupCount=2, encoding=None, delay=False)
# file_fmt = logging.Formatter('%(asctime)s %(name)-5s [%(threadName)-10s] %(levelname)-5s %(funcName)-5s '
#                              '%(filename)s line=%(lineno)-4d: %(message)s')
# file_hdl.setFormatter(file_fmt)
# logger.addHandler(file_hdl)
#
# # set log level
# log_key = 'DEBUG'
# log_lv = {'DEBUG': (logging.DEBUG, logger.debug), 'INFO': (logging.INFO, logger.info)}
# level, lg = log_lv[log_key]
# logger.setLevel(level)
# # logger.debug('often makes a very good meal of %s', 'visiting tourists')
#
# import pickle
#
# """Step 3. Path control
# """
# import warnings
#
# # add 'itod' root directory into sys.path
# root_dir = os.getcwd()
# sys.path.append(root_dir)
# print(f'sys.path: {sys.path}')
# # check the workspace directory for 'open' function.
# # Note this directory can be different with the sys.path root directory
# workspace_dir = os.path.abspath(os.path.join(root_dir))
# if os.path.abspath(os.getcwd()) != workspace_dir:
#     msg = f'current directory does not equal workspace. Changing it to \'{workspace_dir}\'.'
#     warnings.warn(msg)
#     os.chdir(workspace_dir)
# print(f'workspace_dir: {workspace_dir}')

"""Step 4. Constant
"""
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
