import torch
from SimulationData import *
from SimulationDatasetHelper import *
import torchvision.transforms as transforms


class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_list, point_indices, preprocessing_type, use_real_time, lbllength, ini_label_type):
        self.data_list = data_list.data_list
        self.data_train_list = data_list.data_train_list
        self.data_test_list = data_list.data_test_list
        self.data_path = data_path
        self.transform_output = None
        self.transform_input = None
        self.transform_u = None
        self.transform_standard_u = None
        self.transform_standard_q0 = None
        self.transform_standard_u_scale_only = None
        self.point_indices = point_indices
        self.preprocessing_type = preprocessing_type
        self.use_real_time = use_real_time
        self.lbllength = lbllength
        self.label_priority = None
        self.ini_label_type = ini_label_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.data_list[idx]
        sim_data = SimulationState(filename, point_indices=self.point_indices)
        filename_ini = obtainInitialStateFilename(filename)
        sim_data_ini = SimulationState(
            filename_ini, point_indices=self.point_indices)

        q = sim_data.q
        q0 = sim_data_ini.q
        u = q - q0

        if self.transform_output:
            exit('invalid transform_output')

        if self.preprocessing_type == 'stan':
            if self.transform_standard_u:
                u = self.transform_standard_u(u)
        elif self.preprocessing_type == 'norm':
            exit('invalid preprocessing_type')

        lbl = np.zeros(self.lbllength)
        if self.use_real_time:
            exit('invalid use_real_time')
        if self.transform_input:
            exit('invalid transform_input')

        data_item = {'filename': sim_data.filename,
                     'q': q, 'q0': q0, 'u': u, 'lbl': lbl}
        return data_item

    def data_locations(self):
        for data_location in self.data_list:
            yield data_location

    def data_locations_train(self):
        for data_location in self.data_train_list:
            yield data_location

    def data_locations_test(self):
        for data_location in self.data_test_list:
            yield data_location


class SimulationSnapshot(torch.utils.data.Dataset):
    def __init__(self, data_location, transform_output, transform_u, transform_standard_u, transform_standard_q0, transform_standard_u_scale_only, transform_input, point_indices, preprocessing_type, use_real_time, lbllength, label_priority, ini_label_type):
        self.data_location = data_location
        self.transform_output = transform_output
        self.transform_u = transform_u
        self.transform_standard_u = transform_standard_u
        self.transform_standard_q0 = transform_standard_q0
        self.transform_standard_u_scale_only = transform_standard_u_scale_only
        self.transform_input = transform_input
        self.data_location_ini = obtainInitialStateFilename(
            self.data_location)
        self.id_con = []
        self.id_nbr = [] # neighrbor id of the dataset
        self.id_con_nbr = [] # neighbor id of the hidden quant
        self.filename_nbr = []
        self.point_indices = point_indices
        self.preprocessing_type = preprocessing_type
        self.use_real_time = use_real_time
        self.lbllength = lbllength
        self.label_priority = label_priority
        self.ini_label_type = ini_label_type
        self.distance0 = []

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.data_location
        sim_data = SimulationState(filename, point_indices=self.point_indices)
        filename_ini = self.data_location_ini
        sim_data_ini = SimulationState(
            filename_ini, point_indices=self.point_indices)
        q0 = sim_data_ini.q

        q = sim_data.q
        v = sim_data.v
        u = q - q0
        q0_original = q0
        if self.transform_output:
            q = self.transform_output(q)
            q0 = self.transform_output(q0)

        if self.preprocessing_type == 'stan':
            if self.transform_standard_u:
                u = self.transform_standard_u(u)
            if self.transform_standard_u_scale_only:    
                v = self.transform_standard_u_scale_only(v)
        elif self.preprocessing_type == 'norm':
            exit('invalid preprocessing_type')

        lbl = np.zeros(self.lbllength)
        if self.use_real_time:
            exit('invalid use_real_time')
        if self.transform_input:
            lbl = self.transform_input(lbl)

        f_tensor = torch.from_numpy(sim_data.f_tensor)

        time_nbr = sim_data.time
        u_nbr = u
        v_nbr = v
        if self.filename_nbr != []:
            sim_data_nbr = SimulationState(
                self.filename_nbr, point_indices=self.point_indices)
            time_nbr = sim_data_nbr.time
            q_nbr = sim_data_nbr.q
            u_nbr = q_nbr - q0_original
            v_nbr = sim_data_nbr.v
            if self.preprocessing_type == 'stan':
                if self.transform_standard_u:
                    u_nbr = self.transform_standard_u(u_nbr)
                if self.transform_standard_u_scale_only:
                    v_nbr = self.transform_standard_u_scale_only(v_nbr)
            elif self.preprocessing_type == 'norm':
                if self.transform_u:
                    u_nbr = self.transform_u(u_nbr)
        else:
            self.filename_nbr = sim_data.filename

        data_item = {'filename': sim_data.filename,
                     'q': q, 'u': u, 'lbl': lbl, 'q0': q0, 'q0_original': q0_original, 'f_tensor': f_tensor, 'id_con': self.id_con, 'id_nbr': self.id_nbr, 'time': sim_data.time, 'time_nbr': time_nbr, 'u_nbr': u_nbr, 'filename_nbr':self.filename_nbr, 'v': v, 'v_nbr': v_nbr, 'distance0': self.distance0, 'id_con_nbr': self.id_con_nbr}
        return data_item

    def assignHidLayer(self, id_con, lbl_hid):
        self.id_con = id_con
        self.lbl_hid = lbl_hid
    
    def assignDistance0(self, distance0):
        self.distance0 = distance0

    def assignNbrId(self, id_nbr, filename_nbr):
        self.id_nbr = id_nbr
        self.filename_nbr = filename_nbr
    
    def assignNbrHidLayer(self, id_con_nbr):
        self.id_con_nbr = id_con_nbr
