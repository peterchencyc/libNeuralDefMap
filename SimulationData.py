import h5py
import torch
import os
import numpy as np


class SimulationState(object):
    def __init__(self, filename, readfile=True, input_q=None, point_indices=[], input_f_tensor=None, input_v_tensor=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.q = h5_file['/q'][:]
                self.q = self.q[:, point_indices]
                f_tensor_col_major = h5_file['/f_tensor'][:]
                f_tensor_col_major = f_tensor_col_major[:, point_indices]
                # obtain n by dim matrix
                self.q = self.q.T
                f_tensor_col_major = f_tensor_col_major.T
                self.f_tensor = f_tensor_col_major.reshape(
                    -1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
                if '/time' in h5_file:
                    self.time = h5_file['/time'][0][0]
                if '/q_prime' in h5_file:
                    self.v = h5_file['/q_prime'][:]
                    self.v = self.v[:, point_indices]
                    self.v = self.v.T
                if '/q_b4proj' in h5_file:
                    self.q_b4proj = h5_file['/q_b4proj'][:]
                    self.q_b4proj = self.q_b4proj[:, point_indices]
                    self.q_b4proj = self.q_b4proj.T
                if '/q_prime_b4proj' in h5_file:
                    self.v_b4proj = h5_file['/q_prime_b4proj'][:]
                    self.v_b4proj = self.v_b4proj[:, point_indices]
                    self.v_b4proj = self.v_b4proj.T
                if '/f_tensor_b4proj' in h5_file:
                    f_tensor_b4proj_col_major = h5_file['/f_tensor_b4proj'][:]
                    f_tensor_b4proj_col_major = f_tensor_b4proj_col_major[:, point_indices]
                    f_tensor_b4proj_col_major = f_tensor_b4proj_col_major.T
                    self.f_tensor_b4proj = f_tensor_b4proj_col_major.reshape(
                    -1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
                if '/label' in h5_file:
                    self.label = h5_file['/label'][:]
                    if len(self.label.shape) == 2:
                        self.label = self.label[:, 0]
                        

        else:
            if input_q is None:
                print('must provide a q if not reading from file')
                exit()
            self.q = input_q
            if input_f_tensor is not None:
                self.f_tensor = input_f_tensor
            if input_v_tensor is not None:
                self.q_prime = input_v_tensor

        self.npoints = self.q.shape[0]
        self.dim = self.q.shape[1]

    def write_to_file(self, filename):
        self.filename = filename
        print('writng sim state: ', filename)
        dirname = os.path.dirname(self.filename)
        os.makedirs(dirname, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("q", data=self.q.T)
            if self.f_tensor is not None:
                f_tensor_col_major = self.f_tensor.reshape(
                    -1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
                dset = h5_file.create_dataset(
                    "f_tensor", data=f_tensor_col_major.T)
            if self.q_prime is not None:
                dset = h5_file.create_dataset("q_prime", data=self.q_prime.T)
            if self.label is not None:
                label = self.label.reshape(-1, 1)
                label = label.astype(np.float64)
                dset = h5_file.create_dataset("label", data=label)
