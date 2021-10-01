import os
import torch
import torch.optim as optim
from DataList import *
from SimulationDataset import *
from Preprocessing import *
from Net import *
from itertools import repeat
from enum import Enum
from torch.autograd import grad
from timeit import default_timer as timer
from collections.abc import Sequence
import math
        
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def generateLR(learning_rates, epochs):
    assert(len(learning_rates)==len(epochs))
    accumulated_epochs = [0]
    accumulated_epochs.extend(np.cumsum(epochs))
    EPOCH_SIZE = accumulated_epochs[-1]
    def adaptiveLRfromRange(epoch):
        for idx in range(len(accumulated_epochs)-1):
            do = accumulated_epochs[idx]
            up = accumulated_epochs[idx+1]
            if do <= epoch < up:
                return learning_rates[idx]
        if epoch == accumulated_epochs[-1]: #last epoch
            return learning_rates[-1]
        else:
            exit('invalid epoch for adaptiveLRfromRange')
            
    return adaptiveLRfromRange, EPOCH_SIZE

def BatchAndEpoch(snap_type):
    if snap_type == 'automapconv':
        return 8, 2500


def generateSnapshot(data_location, transform_output, transform_u, transform_standard_u, transform_standard_q0, transform_standard_u_scale_only, transform_input, point_indices, preprocessing_type, use_real_time, lbllength, label_priority, ini_label_type):
    return SimulationSnapshot(data_location, transform_output, transform_u, transform_standard_u, transform_standard_q0, transform_standard_u_scale_only, transform_input, point_indices, preprocessing_type, use_real_time, lbllength, label_priority, ini_label_type)


def fullDatasetFromMaster(master_dataset):
    snapshots = []
    for data_location in master_dataset.data_locations_train():
        snapshots.append(generateSnapshot(
            data_location, master_dataset.transform_output, master_dataset.transform_u, master_dataset.transform_standard_u, master_dataset.transform_standard_q0, master_dataset.transform_standard_u_scale_only, master_dataset.transform_input, master_dataset.point_indices, master_dataset.preprocessing_type, master_dataset.use_real_time, master_dataset.lbllength, master_dataset.label_priority, master_dataset.ini_label_type))
    train_dataset = torch.utils.data.ConcatDataset(snapshots)
    snapshots = []
    for data_location in master_dataset.data_locations_test():
        snapshots.append(generateSnapshot(
            data_location, master_dataset.transform_output, master_dataset.transform_u, master_dataset.transform_standard_u, master_dataset.transform_standard_q0, master_dataset.transform_standard_u_scale_only, master_dataset.transform_input, master_dataset.point_indices, master_dataset.preprocessing_type, master_dataset.use_real_time, master_dataset.lbllength, master_dataset.label_priority, master_dataset.ini_label_type))
    if len(snapshots) > 0:
        test_dataset = torch.utils.data.ConcatDataset(snapshots)
        full_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, test_dataset])
    else:
        test_dataset = train_dataset
        full_dataset = train_dataset
    return full_dataset, train_dataset, test_dataset


def detectNumberOfPoints(sample_file):
    with h5py.File(sample_file, 'r') as h5_file:
        q = h5_file['/q'][:]
    return list(range(0, q.shape[1]))


def prepareData(data_path, point_indices, preprocessing_type, train_ratio, use_real_time, lbllength, onlyinifilename, ini_label_type, samp_start_number, samp_gap, samp_gap_enc, temporal_sampling_every_n):
    data_list = DataList(data_path, train_ratio, onlyinifilename, temporal_sampling_every_n)
    # print(data_list.data_list)
    # print('data list done')
    # exit()
    if point_indices is None:
        point_indices = detectNumberOfPoints(data_list.data_list[0])
        point_indices = point_indices[samp_start_number::samp_gap]
    
    point_indices_enc = detectNumberOfPoints(data_list.data_list[0])
    point_indices_enc = point_indices_enc[samp_start_number::samp_gap_enc]
        
    assert (len(data_list.data_list) > 0)
    master_dataset = SimulationDataset(
        data_path, data_list, point_indices_enc, preprocessing_type, use_real_time, lbllength, ini_label_type)

    # data normalization
    preprop = Preprocessing(master_dataset)
    transform_output, transform_u = preprop.computeNormalizationTransformation()
    transform_standard_u, transform_standard_q0, transform_standard_u_scale_only = preprop.computeStandardizeTransformation()
    transform_input = preprop.computeLabelStandardizeTransformation(lbllength)

    master_dataset.transform_output = transform_output
    master_dataset.transform_u = transform_u
    master_dataset.transform_standard_u = transform_standard_u
    master_dataset.transform_standard_q0 = transform_standard_q0
    master_dataset.transform_standard_u_scale_only = transform_standard_u_scale_only
    master_dataset.transform_input = transform_input

    # individual frame as a dataset
    full_dataset, train_dataset, test_dataset = fullDatasetFromMaster(
        master_dataset)

    # to get reproducible training (such as loss) by setting torch's initial_seed
    torch.manual_seed(0)

    return full_dataset, train_dataset, test_dataset, preprop, point_indices, point_indices_enc


def prepareOther(snap_type, time_only=False, point_indices=[], lbllength=5):
    # other setups
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if multiple devices, find one with large empty memory
    if device.type == 'cuda':
        nvmlInit()
        device_id = 0
        free_memory_max = 0
        for i in range(torch.cuda.device_count()):
            total_memory_gb_total, total_memory_gb_used, total_memory_gb_free = getGMem(i)
            if total_memory_gb_free > free_memory_max:
                device_id = i
                free_memory_max = total_memory_gb_free
        
        if free_memory_max > 4.0:    
            device = torch.device('cuda:'+str(device_id))
        else:
            exit('not enough cuda memory')
    print('device: ', device)
    print('device free memory: ', free_memory_max)

    if snap_type == 'automapconv':
        net = NetAutoEnc(len(point_indices), lbllength).to(device)
    criterion = nn.MSELoss()

    BATCH_SIZE, EPOCH_SIZE = BatchAndEpoch(snap_type)

    return device, net, criterion, BATCH_SIZE, EPOCH_SIZE

def prepNetworkInputFromXhat(device, batch, lbl):
    q0 = batch['q0'].float()
    npoints = q0.size(2)
    q0 = q0.view(q0.size(0) * q0.size(2), q0.size(1), q0.size(3))
    q0 = q0.view(q0.size(0), -1)

    lbl_empty = None
    for i in range(lbl.size(0)):
        lbl_local = lbl[i, :]
        lbl_local = lbl_local.expand(npoints, lbl_local.size(1))
        if lbl_empty == None:
            lbl_empty = lbl_local
        else:
            lbl_empty = torch.cat((lbl_empty, lbl_local), 0)

    lbl = lbl_empty
    lbl = lbl.to(device)
    q0 = q0.to(device)
    lbl = torch.cat((lbl, q0), 1)

    return lbl

def dataFromBatch(data_path, batch, snap_type, sup_type, hidden_quant, lbllength, device, time_only, inter=None):
    q = batch['q'].float()
    u = batch['u'].float()
    v = batch['v'].float()

    # have to write lbl myself because dataloader doesnt support computation graph
    if snap_type != 'automapconv':
        id_cons = batch['id_con']
        per_snapshot_length = lbllength
        nsample = id_cons.size(0)
        id_cons = per_snapshot_length * id_cons.view(-1, 1)
        id_one = torch.from_numpy(np.linspace(
            0, per_snapshot_length-1, per_snapshot_length, endpoint=True, dtype=int))
        id_one = id_one.view(1, -1)

        id_cons = id_cons.repeat(1, per_snapshot_length)
        id_one = id_one.repeat(id_cons.size(0), 1)
        id_cons += id_one

        id_cons = id_cons.view(id_cons.size(0) * id_cons.size(1))

        lbl = hidden_quant[torch.LongTensor(id_cons)]
        lbl = lbl.view(nsample, -1)

    lbl_mid_list = None
    u_mid_list = None
    
    if snap_type == 'map':
        q0 = batch['q0'].float()
        npoints = q0.size(2)
        q0 = q0.view(q0.size(0) * q0.size(2), q0.size(1), q0.size(3))
        q0 = q0.view(q0.size(0), -1)

        assert(npoints == q.size(2))
        q = q.view(q.size(0) * q.size(2), q.size(1), q.size(3))
        u = u.view(u.size(0) * u.size(2), u.size(1), u.size(3))
        v = v.view(v.size(0) * v.size(2), v.size(1), v.size(3))

        lbl_empty = None
        for i in range(lbl.size(0)):
            lbl_local = lbl[i, :]
            lbl_local = lbl_local.expand(npoints, lbl.size(1))
            if lbl_empty == None:
                lbl_empty = lbl_local
            else:
                lbl_empty = torch.cat((lbl_empty, lbl_local), 0)

        lbl = lbl_empty

        lbl = lbl.to(device)
        q0 = q0.to(device)
        lbl = torch.cat((lbl, q0), 1)

    elif snap_type == 'automapconv':
        q0 = batch['q0'].float()
        npoints = q0.size(2)
        q0 = q0.view(q0.size(0), q0.size(2), q0.size(3))
        u_for_lbl = u.view(u.size(0), u.size(2), u.size(3))

        assert(npoints == q.size(2))
        q = q.view(q.size(0) * q.size(2), q.size(1), q.size(3))
        u = u.view(u.size(0) * u.size(2), u.size(1), u.size(3))
        v = v.view(v.size(0) * v.size(2), v.size(1), v.size(3))

        u_for_lbl = u_for_lbl.to(device)
        q0 = q0.to(device)
        lbl = torch.cat((u_for_lbl, q0), 2)
    q = q.view(q.size(0), -1)
    u = u.view(u.size(0), -1)
    v = v.view(v.size(0), -1)
    return q, u, lbl, v, lbl_mid_list, u_mid_list

def defmapFromBatch(batch, device):
    f_tensor = batch['f_tensor'].float().to(device)
    return f_tensor

def iniHidLabel(dataset, lbllength):
    nsnapshot = len(dataset)
    datasets = dataset.datasets
    hidLabel = torch.zeros(nsnapshot * lbllength, requires_grad=False)
    return hidLabel

def assignHidLabel2Dataset(dataset, hidLabel, lbllength):
    datasets = dataset.datasets  # assuming class ConcatDataset
    for i in range(len(datasets)):
        snapshot = datasets[i]
        snapshot.assignHidLayer(i,
                                hidLabel[i*lbllength:i*lbllength + lbllength])
                                
def buildFilenameIndexMap(dataset_filenames):
    filename_index_map = {}
    for i in range(len(dataset_filenames)):
        filename_index_map[dataset_filenames[i]] = i
    return filename_index_map

class FilenameManager():
    def __init__(self):
        self.increment = None

    def obtainNext(self, filename):
        if self.increment is None:
            exit('set increment ')
        base_name = os.path.basename(filename)
        dir_name = os.path.dirname(filename)
        config_file_match = config_file_matcher.match(
            os.path.basename(base_name))
        t = int(config_file_match[1])
        t_next = t + self.increment
        filename_next = os.path.join(os.path.dirname(
            filename), hprefix + '_' + '{:010d}'.format(t_next)+'.h5')
        return filename_next
        

def assignNbrId(dataset, filename_index_map, dataset_filenames, filename_manager):
    datasets = dataset.datasets  # assuming class ConcatDataset
    for i in range(len(datasets)):
        snapshot = datasets[i]
        snapshot.id_con = i
    for i in range(len(datasets)):
        snapshot = datasets[i]
        filename = dataset_filenames[i]
        filename_next = filename_manager.obtainNext(filename)
        # print(filename_next)
        if filename_next in filename_index_map:
            id_nbr = filename_index_map[filename_next]
            snapshot.assignNbrId(id_nbr, filename_next)
            snapshot_nbr = datasets[id_nbr]
            id_con_nbr = snapshot_nbr.id_con
            snapshot.assignNbrHidLayer(id_con_nbr)
        else:
            id_nbr = -1
            snapshot.assignNbrId(id_nbr, [])
            snapshot.assignNbrHidLayer(-1)
    # exit()

def obtainFilenames(data_path, dataset, train_ratio_dir, error_type='train', filename_manager=None):
    if error_type == 'train':
        filename = 'filenames_inc-' + str(filename_manager.increment) + '.npy'
    elif error_type == 'test':
        filename = 'filenames_test_inc-' + str(filename_manager.increment) + '.npy'

    filename_file = os.path.join(
            data_path, os.path.join(train_ratio_dir, filename))
    if os.path.exists(filename_file):
        # read
        with open(filename_file, 'rb') as f:
            filenames = np.load(f)
    else:    
        filenames = []
        datasets = dataset.datasets  # assuming class ConcatDataset
        for i in range(len(datasets)):
            filenames.append(datasets[i][0]['filename'])
        filenames = np.array(filenames)
        # write
        with open(filename_file, 'wb') as f:
            np.save(f, filenames)
    return filenames

def dydxFromNetGrad(lbl, preprop, net_grad, device):
    outputs_grad = net_grad(lbl)
    dydx_preprocessed = outputs_grad[:, :, -3:]
    dXtildedX = preprop.NormalizationTransformationTorchGrad.view(1, 3, 3).float()
    dXtildedX = dXtildedX.expand(dydx_preprocessed.size(0), 3, 3)
    dudutilde = preprop.InvStandardizeUTorchGrad.view(1, 3, 3).float()
    dudutilde = dudutilde.expand(dydx_preprocessed.size(0), 3, 3)
    dydx = torch.matmul(dydx_preprocessed, dXtildedX)
    dydx = torch.matmul(dudutilde, dydx)
    dydx += torch.eye(3).to(device)
    return dydx

def write(outputs, outputs_f, outputs_v, filename, split_prefix, snap_type,label):
    predicted_sim_state = SimulationState('', False, outputs, [], outputs_f, outputs_v)
    predicted_sim_state.label = label
    writefilename = convertInputFilenameIntoOutputFilename(
        filename, split_prefix, snap_type)
    predicted_sim_state.write_to_file(writefilename
                                      )

def write_data(position_write_list, filename_write_list, deformation_write_list, velocity_dict_pred, velocity_dict_original, split_prefix, snap_type,label_write_list, filename_model):
    for position, filename, deformation, label in zip(position_write_list, filename_write_list, deformation_write_list,label_write_list):
        if filename in velocity_dict_pred.keys():
            velocity = velocity_dict_pred[filename]
        else:
            velocity = velocity_dict_original[filename]
        
        basename0 = os.path.basename(filename)
        basename1 = os.path.basename(os.path.dirname(filename))
        dirname1 = os.path.dirname(os.path.dirname(filename))
        dirname1 += '_' + filename_model
        filename = os.path.join(dirname1, basename1, basename0)
        
        write(position, deformation, velocity, filename, split_prefix, snap_type,label)

# legacy autograd
def computeDefMap(data_path, batch, device, preprop, preprocessing_type, net, invstandardizeUTorch, create_graph, q_backup, time_only, sup_type, hidden_quant, lbllength, lbl_overwrite=None):
    lbl_grad, q0_original = dataFromBatchForDefGrad(
        data_path, batch, device, preprop, time_only, sup_type, hidden_quant, lbllength)
    if lbl_overwrite != None:
        lbl_grad[:, 0:lbl_overwrite.size(1)] = lbl_overwrite
    outputs_grad = net(lbl_grad)
    outputs_grad = outputs_grad.view_as(q_backup)
    if preprocessing_type == 'stan':
        outputs_grad = invstandardizeUTorch(
            outputs_grad.float())
    elif preprocessing_type == 'norm':
        exit(
            'preprocessing_type == \'norm\' not supported for outputs_grad')
    outputs_grad = outputs_grad.view(
        outputs_grad.size(0)*outputs_grad.size(2), -1)

    dy_dx_full_minusI = computeDeformationGradient(
        q0_original, outputs_grad, create_graph)
    dy_dx_full = dy_dx_full_minusI + \
        torch.eye(3).view(1, -1).to(device)
    return dy_dx_full

def dataFromBatchForDefGrad(data_path, batch, device, preprop, time_only, sup_type, hidden_quant, lbllength):
    q0_original = batch['q0_original'].float().clone()
    q0_original = q0_original.to(device)
    q0_original.requires_grad_()
    npoints = q0_original.size(1)
    transform_output_torch = preprop.computeNormalizationTransformationTorch(
        device)
    q0_original_transformed = transform_output_torch(q0_original).float()
    q0_original_transformed = q0_original_transformed.view(
        q0_original_transformed.size(0)*q0_original_transformed.size(1), -1)

    # have to write lbl myself because dataloader doesnt support computation graph
    id_cons = batch['id_con']
    per_snapshot_length = lbllength
    nsample = id_cons.size(0)
    id_cons = per_snapshot_length * id_cons.view(-1, 1)
    id_one = torch.from_numpy(np.linspace(
        0, per_snapshot_length-1, per_snapshot_length, endpoint=True, dtype=int))
    id_one = id_one.view(1, -1)

    id_cons = id_cons.repeat(1, per_snapshot_length)
    id_one = id_one.repeat(id_cons.size(0), 1)
    id_cons += id_one

    id_cons = id_cons.view(id_cons.size(0) * id_cons.size(1))

    lbl = hidden_quant[torch.LongTensor(id_cons)]
    lbl = lbl.view(nsample, -1)

    lbl_empty = None
    for i in range(lbl.size(0)):
        lbl_local = lbl[i, :]
        lbl_local = lbl_local.expand(npoints, lbl.size(1))
        if lbl_empty == None:
            lbl_empty = lbl_local
        else:
            lbl_empty = torch.cat((lbl_empty, lbl_local), 0)
    lbl = lbl_empty
    lbl = lbl.to(device)

    lbl_grad = torch.cat((lbl, q0_original_transformed), 1)

    return lbl_grad, q0_original


def computeDeformationGradient(lbl, outputs, create_graph, effective_length=None):
    dy0_dx = grad(outputs=outputs[:, 0], inputs=lbl, grad_outputs=torch.ones_like(outputs[:, 0]),
                  retain_graph=True, create_graph=create_graph)[0]
    dy1_dx = grad(outputs=outputs[:, 1], inputs=lbl, grad_outputs=torch.ones_like(outputs[:, 1]),
                  retain_graph=True, create_graph=create_graph)[0]
    dy2_dx = grad(outputs=outputs[:, 2], inputs=lbl, grad_outputs=torch.ones_like(outputs[:, 2]),
                  retain_graph=True, create_graph=create_graph)[0]
    if effective_length:
        dy0_dx = dy0_dx[:, 0:effective_length]
        dy1_dx = dy1_dx[:, 0:effective_length]
        dy2_dx = dy2_dx[:, 0:effective_length]

    dy_dx_full = torch.cat((dy0_dx, dy1_dx, dy2_dx), -1)
    return dy_dx_full