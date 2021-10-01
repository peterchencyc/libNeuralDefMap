from SimulationDatasetHelper import *


class DataList(object):
    def __init__(self, root_dir, train_ratio, onlyinifilename, temporal_sampling_every_n):
        self.data_list, self.data_train_list, self.data_test_list, self.data_train_dir, self.data_test_dir = obtainFilesRecursively(
            root_dir, train_ratio, onlyinifilename, temporal_sampling_every_n)
