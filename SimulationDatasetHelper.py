import re
import glob
import os
import numpy as np
import random
import math

hprefix = 'h5_f'
config_file_pattern = r'h5_f_(\d+)\.h5'
config_file_matcher = re.compile(config_file_pattern)
dir_pattern = r'sim_seq_(.*?)'
dir_matcher = re.compile(dir_pattern)

def dropThisIniFile(fullfilename, onlyinifilename):
    if onlyinifilename is not None:
        config_file_match = config_file_matcher.match(
                os.path.basename(fullfilename))
        if int(config_file_match[1]) == 0:
            if not (fullfilename==onlyinifilename):
                return True
    return False


def obtainFilesRecursively(path, train_ratio, onlyinifilename, temporal_sampling_every_n=1):
    data_list = []
    data_train_list = []
    data_test_list = []
    data_train_dir = []
    data_test_dir = []

    dir_list = os.listdir(path)

    num_sims = 0
    dir_list_sim = []
    for dirname in dir_list:
        dir_match = dir_matcher.match(
            dirname)
        if dir_match != None:
            num_sims += 1
            dir_list_sim.append(dirname)

    random.seed(0)
    random.shuffle(dir_list_sim)

    train_size = math.ceil(train_ratio * num_sims)
    test_size = num_sims - train_size

    counter = 0
    for dirname in dir_list_sim:
        data_list_local = data_train_list if counter < train_size else data_test_list
        data_dir_local = data_train_dir if counter < train_size else data_test_dir
        data_dir_local.append(os.path.join(path, dirname))
        counter += 1
        for filename in os.listdir(os.path.join(path, dirname)):
            config_file_match = config_file_matcher.match(
                filename)
            if config_file_match is None:
                continue
            # skip files begin
            file_number = int(config_file_match[1])
            if file_number%temporal_sampling_every_n!=0:
                continue
            # skip files finish
            # print(file_number)
            fullfilename = os.path.join(path, dirname, filename)
            if not dropThisIniFile(fullfilename, onlyinifilename):
                data_list.append(fullfilename)
                data_list_local.append(fullfilename)
        # exit()
    return data_list, data_train_list, data_test_list, data_train_dir, data_test_dir


def convertInputFilenameIntoOutputFilename(filename_in, split_prefix, snap_type):
    config_file_match = config_file_matcher.match(
        os.path.basename(filename_in))

    dirname = os.path.dirname(filename_in)
    pardirname = os.path.dirname(dirname)
    pardirname += '_pred-'+snap_type
    dirname = os.path.basename(dirname)
    dirname += '_pred'

    return os.path.join(pardirname, split_prefix, dirname) + '/'+hprefix+'_'+config_file_match[1]+'.h5'


def obtainInitialStateFilename(filename):
    return os.path.join(os.path.dirname(filename),
                        hprefix + '_' + ''.zfill(10))+'.h5'
