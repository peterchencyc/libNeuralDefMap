import torchvision.transforms as transforms
import multiprocessing as mp
import numpy as np
import torch
import os
import h5py

class Preprocessing(object):
    def __init__(self, sim_dataset):
        self.sim_dataset = sim_dataset
        [self.npoints, self.dim] = self.sim_dataset[0]['q'].shape
        self.min_global = np.inf*np.ones(self.dim)
        self.max_global = -self.min_global
        self.min_global_u = np.inf*np.ones(self.dim)
        self.max_global_u = -self.min_global_u
        self.mean_lbl = None
        self.std_lbl = None

    def fU(self, idx):
        data_item = self.sim_dataset[idx]
        u = data_item['u']
        return np.amin(u, axis=0), np.amax(u, axis=0)
    
    def computeMinAndMaxParallelU(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_u.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.min_global_u = np.load(f)
                self.max_global_u = np.load(f)
        else:
            with mp.Pool(processes=mp.cpu_count()) as p:
                results = p.map(self.fU, range(len(self.sim_dataset)))
                for min_local, max_local in results:
                    self.min_global_u = np.minimum(
                        self.min_global_u, min_local)
                    self.max_global_u = np.maximum(
                        self.max_global_u, max_local)
            # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_global_u)
                np.save(f, self.max_global_u)

    def f(self, idx):
        data_item = self.sim_dataset[idx]
        q = data_item['q']
        return np.amin(q, axis=0), np.amax(q, axis=0)

    def computeMinAndMaxParallel(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.min_global = np.load(f)
                self.max_global = np.load(f)
        else:
            with mp.Pool(processes=mp.cpu_count()) as p:
                results = p.map(self.f, range(len(self.sim_dataset)))
                for min_local, max_local in results:
                    self.min_global = np.minimum(self.min_global, min_local)
                    self.max_global = np.maximum(self.max_global, max_local)
            # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_global)
                np.save(f, self.max_global)

    def normalize(self, q):
        return (q - self.min_global_points) / self.dist_global_points

    def normalizeTorch(self, q):
        return (q - self.min_global_points_torch) / self.dist_global_points_torch

    def normalizeU(self, q):
        return (q - self.min_global_points_u) / self.dist_global_points_u

    def computeNormalizationTransformation(self):
        self.computeMinAndMaxParallel()
        print('global bb min: ', self.min_global)
        print('global bb max: ', self.max_global)
        self.dist_global = self.max_global-self.min_global
        print('global bb dis: ', self.dist_global)
        self.min_global_points = np.tile(self.min_global, (1, 1))
        self.max_global_points = np.tile(self.max_global, (1, 1))
        self.dist_global_points = self.max_global_points - self.min_global_points

        self.computeMinAndMaxParallelU()
        print('u bb min: ', self.min_global_u)
        print('u bb max: ', self.max_global_u)
        self.min_global_points_u = np.tile(
            self.min_global_u, (1, 1))
        self.max_global_points_u = np.tile(
            self.max_global_u, (1, 1))
        self.dist_global_points_u = self.max_global_points_u - self.min_global_points_u

        return transforms.Compose(
            [transforms.Lambda(self.normalize), transforms.ToTensor()]), transforms.Compose(
            [transforms.Lambda(self.normalizeU), transforms.ToTensor()])

    def computeNormalizationTransformationTorch(self, device):
        self.min_global_points_torch = torch.from_numpy(
            self.min_global_points).to(device)
        self.max_global_points_torch = torch.from_numpy(
            self.max_global_points).to(device)
        self.dist_global_points_torch = torch.from_numpy(
            self.dist_global_points).to(device)
            
        grad_vec = torch.div(torch.ones_like(self.dist_global_points_torch[0,:]), self.dist_global_points_torch[0,:])
        self.NormalizationTransformationTorchGrad = torch.diag(grad_vec)
        return transforms.Compose(
            [transforms.Lambda(self.normalizeTorch)])

    def invnormalizeU(self, u_normalized):
        return self.min_global_points_u + np.multiply(u_normalized, self.dist_global_points_u)

    def computeInvNormalizationTransformationU(self):
        return transforms.Compose(
            [transforms.Lambda(self.invnormalizeU)])

    def g(self, idx):
        data_item = self.sim_dataset[idx]
        lbl = data_item['lbl']
        return lbl
    
    def computeMeanAndStd(self, lbllength):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_lbl-' + str(lbllength) + '.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.mean = np.load(f)
                self.std = np.load(f)
        else:
            with mp.Pool(processes=mp.cpu_count()) as p:
                lbls = p.map(self.g, range(len(self.sim_dataset)))
                lbls = np.array(lbls)
                self.mean = np.mean(lbls, axis=0)
                self.std = np.std(lbls, axis=0)
            # process stage with zero std
            for i in range(len(self.std)):
                if self.std[i] < 1e-12:
                    self.std[i] = 1
            # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean)
                np.save(f, self.std)
        self.mean_torch = torch.from_numpy(self.mean)
        self.std_torch = torch.from_numpy(self.std)

    def standardize(self, lbl):
        # https://en.wikipedia.org/wiki/Feature_scaling
        return (lbl - self.mean) / self.std

    def standardizeToTensor(self, lbl):
        return torch.from_numpy(self.standardize(lbl))

    def computeLabelStandardizeTransformation(self, lbllength):
        self.computeMeanAndStd(lbllength)
        print('label length: ', len(self.mean))
        print('label mean: ', self.mean)
        print('label std: ', self.std)

        return self.standardizeToTensor

    def invStandardizeTorch(self, lbl_standardized):
        return self.mean_torch + lbl_standardized * self.std_torch

    def gU(self, idx):
        data_item = self.sim_dataset[idx]
        u = data_item['u']
        return u

    def computeMeanAndStdU(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_u.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.mean_u = np.load(f)
                self.std_u = np.load(f)
        else:
            with mp.Pool(processes=mp.cpu_count()) as p:
                us = p.map(self.gU, range(len(self.sim_dataset)))
                us = np.vstack(us)
                self.mean_u = np.mean(us, axis=0)
                self.std_u = np.std(us, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_u)
                np.save(f, self.std_u)

    def standardizeU(self, u):
        # https://en.wikipedia.org/wiki/Feature_scaling
        return (u - self.mean_u_points) / self.std_u_points
    
    def standardizeUScaleOnly(self, u):
        # https://en.wikipedia.org/wiki/Feature_scaling
        return u / self.std_u_points
    
    def gQ0(self, idx):
        data_item = self.sim_dataset[idx]
        q0 = data_item['q0']
        return q0

    def computeMeanAndStdQ0(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_q0.npy')
        if os.path.exists(preprocessed_file):
            # read
            with open(preprocessed_file, 'rb') as f:
                self.mean_q0 = np.load(f)
                self.std_q0 = np.load(f)
        else:
            with mp.Pool(processes=mp.cpu_count()) as p:
                q0s = p.map(self.gQ0, range(len(self.sim_dataset)))
                q0s = np.vstack(q0s)
                self.mean_q0 = np.mean(q0s, axis=0)
                self.std_q0 = np.std(q0s, axis=0)

            # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_q0)
                np.save(f, self.std_q0)

    def standardizeQ0(self, q0):
        # https://en.wikipedia.org/wiki/Feature_scaling
        return (q0 - self.mean_q0_points) / self.std_q0_points

    def computeStandardizeTransformation(self):
        self.computeMeanAndStdU()
        print('u mean: ', self.mean_u)
        print('u std: ', self.std_u)
        self.mean_u_points = self.mean_u
        self.std_u_points = self.std_u

        self.computeMeanAndStdQ0()
        print('q0 mean: ', self.mean_q0)
        print('q0 std: ', self.std_q0)
        self.mean_q0_points = np.tile(
            self.mean_q0, (1, 1))
        self.std_q0_points = np.tile(
            self.std_q0, (1, 1))

        return transforms.Compose(
            [transforms.Lambda(self.standardizeU), transforms.ToTensor()]), transforms.Compose(
            [transforms.Lambda(self.standardizeQ0), transforms.ToTensor()]), transforms.Compose(
            [transforms.Lambda(self.standardizeUScaleOnly), transforms.ToTensor()])

    def invStandardizeU(self, u_standardized):
        return self.mean_u_points + np.multiply(u_standardized, self.std_u_points)

    def computeInvStandardizeU(self):
        return transforms.Compose(
            [transforms.Lambda(self.invStandardizeU)])

    def invStandardizeUScaleOnly(self, u_standardized):
        return np.multiply(u_standardized, self.std_u_points)

    def computeInvStandardizeUScaleOnly(self):
        return transforms.Compose(
            [transforms.Lambda(self.invStandardizeUScaleOnly)])

    def invStandardizeUTorch(self, u_standardized):
        return self.mean_u_points_torch + u_standardized * self.std_u_points_torch

    def computeInvStandardizeUTorch(self, device):
        self.mean_u_points_torch = torch.from_numpy(
            self.mean_u_points).to(device)
        self.std_u_points_torch = torch.from_numpy(
            self.std_u_points).to(device)
        self.InvStandardizeUTorchGrad = torch.diag(self.std_u_points_torch)
        return transforms.Compose(
            [transforms.Lambda(self.invStandardizeUTorch)])

    def invStandardizeQ0(self, q0standardized):
        return self.mean_q0_points + np.multiply(q0_standardized, self.std_q0_points)

    def computeInvStandardizeQ0(self):
        return transforms.Compose(
            [transforms.Lambda(self.invStandardizeQ0)])

    def saveInfo(self, filename):
        print('writng preprocessing info: ', filename)
        with h5py.File(filename, 'w') as h5_file:
            h5_file.create_dataset("lbl_mean", data=self.mean.reshape(-1, 1))
            h5_file.create_dataset("lbl_std", data=self.std.reshape(-1, 1))
            h5_file.create_dataset(
                "pos_min", data=self.min_global.reshape(-1, 1))
            h5_file.create_dataset(
                "pos_dis", data=self.dist_global.reshape(-1, 1))
            h5_file.create_dataset("mean_u", data=self.mean_u.reshape(-1, 1))
            h5_file.create_dataset("std_u", data=self.std_u.reshape(-1, 1))
