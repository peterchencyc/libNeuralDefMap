from torch.utils.data import Dataset, DataLoader
from PrepareDataset import *

import argparse
import math

from timeit import default_timer as timer

import pathlib

import coverage

cov = coverage.Coverage()
cov.start()

parser = argparse.ArgumentParser(
    description='ROM testing')
parser.add_argument('-d', help='path to the dataset',
                    type=str, nargs=1, required=True)
parser.add_argument('-m', help='path to the saved weights',
                    type=str, nargs=1, required=True)
parser.add_argument('-st', help='network structure \'automapconv\'',
                    type=str, nargs=1, required=False)
parser.add_argument('-disp', help='position type \'u\'',
                    type=str, nargs=1, required=False)
parser.add_argument('-sup', help='supervision type \'full\'',
                    type=str, nargs=1, required=False)
parser.add_argument('-np', help='number of material points used for training',
                    type=int, nargs=1, required=False)
parser.add_argument('-pp', help='preprocessing type \'stan\'. \'stan\' is mean-std standardization',
                    type=str, nargs=1, required=False)
parser.add_argument('-et', help='error type \'test\' or \'train\'',
                    type=str, nargs=1, required=True)
parser.add_argument('-cl', help='compute label',
                    dest='cl', action='store_true')
parser.add_argument('-lbl', help='label length',
                    type=int, nargs=1, required=True)
parser.add_argument('-tr', help='compute label',
                    dest='tr', action='store_true')
parser.add_argument('-train_ratio', help='train /  test splitting',
                    type=float, nargs=1, required=True)
parser.add_argument('-ini_label_file', help='file for computing initial label',
                    type=str, nargs=1, required=False)
parser.add_argument('-end_label_file', help='file for computing initial label',
                    type=str, nargs=1, required=False)
parser.add_argument('-use_special_label', dest='use_special_label',
                    action='store_true')
parser.add_argument('-use_actual_v', dest='use_actual_v',
                    action='store_true')
parser.add_argument('-use_ini_solve', dest='use_ini_solve',
                    action='store_true')
parser.add_argument('-onlyinifilename', type=str, nargs=1, required=False)
parser.add_argument('-ini_label_type', type=str, nargs=1, required=False)
parser.add_argument('-target_dirname', type=str, nargs=1, required=False)
parser.add_argument('-recover', dest='recover',
                    action='store_true')
parser.add_argument('-printlog', dest='printlog',
                    action='store_true')
parser.add_argument('-printxhat', dest='printxhat',
                    action='store_true')
parser.add_argument('-samp_start_number', help='samp_start_number',
                    type=int, nargs=1, required=False)
parser.add_argument('-samp_gap', help='samp_gap',
                    type=int, nargs=1, required=False)
parser.add_argument('-pos_offset', help='pos_offset that roughly moves the sim data to origin; otherwise, the relative position error is meaningless',
                    type=float, nargs=1, required=False)
parser.add_argument('-temporal_sampling_every_n', help='temporal_sampling_every_n',
                    type=int, nargs=1, required=False)
args = parser.parse_args()

position_offset = args.pos_offset[0] if args.pos_offset else 5
position_offset = np.array([position_offset, position_offset, position_offset])

if args.onlyinifilename:
    onlyinifilename = args.onlyinifilename[0]
else:
    onlyinifilename = None

# npyfile, filename
ini_label_type = 'none'

temporal_sampling_every_n = args.temporal_sampling_every_n[0] if args.temporal_sampling_every_n else 1
filename_manager = FilenameManager()
filename_manager.increment = temporal_sampling_every_n

if args.target_dirname:
    target_dirname = args.target_dirname[0]
else:
    target_dirname = None

LBLLENGTH = args.lbl[0]
samp_start_number = args.samp_start_number[0] if args.samp_start_number else 0
samp_gap = args.samp_gap[0] if args.samp_gap else 1
snap_type = 'automapconv'

disp_type = 'u'

sup_type = 'full'
preprocessing_type = 'stan'

error_type = args.et[0]
if error_type != 'train' and error_type != 'test':
    exit('invalid error_type')

data_path = args.d[0]
time_only = False
compute_label = args.cl
compare_def_map = True
create_graph = False
compute_generalized_velocity = True
use_real_time = False

# use 1.0 if want to use all the data for training
train_ratio = args.train_ratio[0]
if train_ratio > 1.0 or train_ratio < 0.0:
    exit('invalid train_ratio')

if args.ini_label_file:
    ini_label_file = args.ini_label_file[0]
    if os.path.exists(ini_label_file) == False:
        exit('ini_label_file does not exist')
else:
    ini_label_file = None

if args.end_label_file:
    end_label_file = args.end_label_file[0]
    if os.path.exists(end_label_file) == False:
        exit('end_label_file does not exist')
else:
    end_label_file = None

if args.np:
    number_points = args.np[0]
else:
    number_points = -1

if number_points == -1:
    point_indices = None
else:
    point_indices = list(range(0, number_points))
samp_gap_enc = samp_gap
full_dataset, train_dataset, test_dataset, preprop, point_indices, point_indices_enc = prepareData(
    data_path, point_indices, preprocessing_type, train_ratio, use_real_time, LBLLENGTH, onlyinifilename, ini_label_type, samp_start_number, samp_gap, samp_gap_enc, temporal_sampling_every_n)
device, net, criterion, BATCH_SIZE, _ = prepareOther(
    snap_type, time_only, point_indices_enc, LBLLENGTH)
criterion_sum = nn.MSELoss(reduction='sum')
invtransformU = preprop.computeInvNormalizationTransformationU()
invstandardizeU = preprop.computeInvStandardizeU()
invstandardizeUScaleOnly = preprop.computeInvStandardizeUScaleOnly()
invstandardizeQ0 = preprop.computeInvStandardizeQ0()
invstandardizeUTorch = preprop.computeInvStandardizeUTorch(device)


if error_type == 'test':
    dataset = test_dataset
elif error_type == 'train':
    dataset = train_dataset

PATH = args.m[0]
net.load_state_dict(torch.load(PATH, map_location=device))
net.eval()

filename_model, _ = os.path.splitext(PATH)
current_dir = pathlib.Path(__file__).parent.absolute()

use_actual_v = args.use_actual_v
use_special_label = True
use_ini_solve = args.use_ini_solve
recover = args.recover
train_ratio_dir = 'train_ratio-' + str(train_ratio)
dataset_filenames = obtainFilenames(data_path, dataset, train_ratio_dir, error_type, filename_manager)

if snap_type != 'automapconv':
    exit('invalid snap_type')
else:
    hidden_quant = iniHidLabel(dataset, LBLLENGTH)
    assignHidLabel2Dataset(dataset, hidden_quant, LBLLENGTH)

if snap_type == 'automapconv':
    preprop.mean = np.zeros_like(preprop.mean)
    preprop.std = np.ones_like(preprop.mean)
    preprop.mean_torch = torch.from_numpy(preprop.mean)
    preprop.std_torch = torch.from_numpy(preprop.std)
    print('preprop.mean set to zero')
    print('preprop.std set to one')

if snap_type == 'automapconv':
    net_enc = NetAutoEncEnc(net)
    net_dec = NetAutoDec(net)
    datasets = dataset.datasets  # assuming class ConcatDataset
    filename_index_map = {}
    # rewrite this using dataloader
    for i in range(len(datasets)):
        snapshot = datasets[i]
        data_item = snapshot[0]
        u = data_item['u'].float()
        u_for_lbl = u.to(device)
        xhat = net_enc(u_for_lbl)
        assert(xhat.size(0)==1)
        snapshot.lbl_hid[:] = xhat[0, :]
        filename_index_map[data_item['filename']] = xhat
        snapshot.point_indices = point_indices
    # can drop the encoder from now on and assuming we are doing automap
    net = net_dec
    snap_type = 'map'

    for filename in sorted(filename_index_map.keys()):
        print(filename)
        print(filename_index_map[filename])

hidden_quant_test = hidden_quant.to(device) # must be called after snapshot.lbl_hid are updated

# compute generalized velocity
if compute_generalized_velocity:
    filename_index_map = buildFilenameIndexMap(dataset_filenames)
    assignNbrId(dataset, filename_index_map, dataset_filenames, filename_manager)

testloader = FastDataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=64)

npoints = len(point_indices)

if args.tr:
    # prepare input for tracing
    data = next(iter(testloader))

    q, u, lbl, _, _, _ = dataFromBatch(
        data_path, data, snap_type, sup_type, hidden_quant_test, LBLLENGTH, device, time_only)

    lbl = lbl.to(torch.device('cpu'))
    net = net.to(torch.device('cpu'))

    start = timer()
    output = net(lbl)
    print('original inference speed: ', timer() - start)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(
            net, lbl, check_trace=True, check_tolerance=1e-20)

    start = timer()
    output = traced_script_module.forward(lbl)
    print('traced inference speed: ', timer() - start)

    dirname_model=filename_model + os.path.basename(os.path.dirname(ini_label_file))
    if use_ini_solve:
        dirname_model += '_inisolved'
    else:
        dirname_model += '_iniunsolved'
    os.makedirs(dirname_model, exist_ok=True)
    dir_filename_model = dirname_model + '/' + filename_model
    traced_script_module.save(dir_filename_model + '.pt')
    preprop.saveInfo(dir_filename_model + '.info')

    # write initial label
    initial_label_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=64) # needed this because FastDataLoader drop_last
    net = net.to(device)
    for batch in initial_label_loader:
        q, u, lbl, _, _, _ = dataFromBatch(data_path, batch, snap_type, sup_type, hidden_quant_test, LBLLENGTH, device, time_only)
        q = q.to(device)
        u = u.to(device)
        lbl = lbl.to(device)

        filenames = batch['filename']
        contain_initial_file = False
        idx_ini = 0
        for idx, filename in enumerate(filenames):
            if filename == ini_label_file:
                contain_initial_file = True
                idx_ini = idx
            if contain_initial_file:
                print(filenames[idx_ini])
                batch_size_local = len(filenames)
                lbl = lbl[idx_ini * npoints:(idx_ini + 1) * npoints,:]
                u = u[idx_ini * npoints:(idx_ini + 1) * npoints,:]
                q = q[idx_ini * npoints:(idx_ini + 1) * npoints,:]
                print('original processed initial generalized coordinate: ', lbl[0, 0:LBLLENGTH])
                if not use_ini_solve:
                    lbl_standardized = lbl[0, 0:LBLLENGTH]
                else:
                    if disp_type == 'u':
                        lbl[:], lbl_alone = leastSquareLbl(
                            lbl, u, net, 1, LBLLENGTH, invstandardizeUTorch)
                    else:
                        exit('invalid disp_type')
                    print('processed initial generalized coordinate: ', lbl_standardized)
                break
    lbl_standardized = lbl_standardized.to(torch.device('cpu'))
    
    lbl = preprop.invStandardizeTorch(lbl_standardized)
    print('unprocessed initial generalized coordinate: ', lbl)

    class TensorContainer(nn.Module):
        def __init__(self, tensor_dict):
            super().__init__()
            for key, value in tensor_dict.items():
                setattr(self, key, value)

    tensor_dict = {'lbl': lbl}
    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)
    tensors.save(dir_filename_model + '.ini_label')
    print('Traced location: ', dir_filename_model)
    exit('Tracing Finished')

if not args.printlog:
    logname = filename_model + '.test_log_' + error_type
    logname = os.path.join(current_dir, logname)
    print('logging to: ', logname)
    logfile = open(logname, 'w')
    sys.stdout = logfile
    sys.stderr = logfile

avg_error = 0.0
running_error_sum = 0.0
running_norm_sum = 0.0

running_loss_raw = 0.0
running_loss_sum_raw = 0.0
running_norm_sum_raw = 0.0

running_loss_raw_q = 0.0
running_loss_sum_raw_q = 0.0
running_norm_sum_raw_q = 0.0

running_def_loss_sum_raw = 0.0
running_def_norm_sum_raw = 0.0

running_def_loss_sum_raw_minusI = 0.0
running_def_norm_sum_raw_minusI = 0.0

running_vel_loss_sum_raw = 0.0
running_vel_norm_sum_raw = 0.0

position_write_list = []
filename_write_list = []
deformation_write_list = []
velocity_dict_pred = {}
velocity_dict_original = {}
label_write_list = []
split_prefix=''

for idx_batch, batch in enumerate(testloader):
    q_backup = batch['q'].float().detach()
    q0_original_backup = batch['q0_original'].float().detach()

    q, u, lbl, v, _, _ = dataFromBatch(
        data_path, batch, snap_type, sup_type, hidden_quant_test, LBLLENGTH, device, time_only)

    q = q.to(device)
    u = u.to(device)
    v = v.to(device)                            
    lbl = lbl.to(device)

    if sup_type == 'semi':
        if error_type == 'test' or compute_label:
            batch_size_local = len(batch['filename'])
            if disp_type == 'u':
                lbl[:], _ = leastSquareLbl(
                    lbl, u, net, batch_size_local, LBLLENGTH, invstandardizeUTorch)
            elif disp_type == 'q':
                lbl[:], _ = leastSquareLbl(
                    lbl, q, net, batch_size_local, LBLLENGTH, invstandardizeUTorch)

    if compare_def_map:
        lbl_overwrite = lbl[:, 0:LBLLENGTH]
        dy_dx_full = computeDefMap(
            data_path, batch, device, preprop, preprocessing_type, net, invstandardizeUTorch, create_graph, q_backup, time_only, sup_type, hidden_quant, LBLLENGTH, lbl_overwrite).detach()
        dy_dx_full_minusI = dy_dx_full - torch.eye(3).view(1, -1).to(device)

        f_tensor = defmapFromBatch(batch, device)
        f_tensor_minusI = f_tensor - torch.eye(3).view(1, -1).to(device)

        zero = torch.zeros_like(f_tensor)
        zero = zero.to(f_tensor)
        running_def_loss_sum_raw += criterion_sum(dy_dx_full, f_tensor)
        running_def_norm_sum_raw += criterion_sum(zero, f_tensor)
        running_def_loss_sum_raw_minusI += criterion_sum(
            dy_dx_full_minusI, f_tensor_minusI)
        running_def_norm_sum_raw_minusI += criterion_sum(
            zero, f_tensor_minusI)

    if compute_generalized_velocity:
        # compute dgdx
        batch_size_local = len(batch['filename'])
        if sup_type == 'full':
            lbl = lbl.detach()
            lbl = lbl.to(device)
            lbl.requires_grad_(True)
        outputs = net(lbl)
        dgdx = computeDeformationGradient(
            lbl, outputs, create_graph, LBLLENGTH)
        dgdx = dgdx.view(batch_size_local, -1, dgdx.size(1))
        dgdx = dgdx.view(batch_size_local, -1, int(dgdx.size(2) / 3))

        u_batch = u.view(batch_size_local, -1, u.size(1))
        v_batch = v.view(batch_size_local, -1, v.size(1))

        # compute dgdt thru dxdt
        filenames = batch['filename']
        times = batch['time']
        id_cons = batch['id_con']
        id_nbrs = batch['id_nbr']
        time_nbrs = batch['time_nbr']
        v_nbrs = batch['v_nbr']
        vs = batch['v']

        for i in range(len(id_cons)):
            filename = filenames[i]
            time = times[i]
            index = id_cons[i]
            index_nbr = id_nbrs[i]
            u_local = u_batch[i]
            v_idx = v_batch[i]

            if target_dirname:
                dirname = os.path.basename(os.path.dirname(filename))
                if dirname != target_dirname:
                    continue
                config_file_match = config_file_matcher.match(
                    os.path.basename(filename))
                print('frame id: ', int(config_file_match[1]))

            label = hidden_quant_test[index *
                                      LBLLENGTH:index * LBLLENGTH + LBLLENGTH]
            label_write_list.append(label.detach().cpu().numpy())
            v_local = vs[i].float().to(
                                device)
            v_local_unprocessed = invstandardizeUScaleOnly(v_local.view(-1, v_local.size(2)).detach().cpu().numpy())
            velocity_dict_original[filename] = v_local_unprocessed.reshape(-1, v_local_unprocessed.shape[1])

            if index_nbr > -1:
                snapshot_nbr = testloader.dataset.datasets[index_nbr]
                snapshot_nbr_item = snapshot_nbr[0]
                filename_nbr = snapshot_nbr_item['filename']
                time_nbr = time_nbrs[i].item()
                label_nbr = hidden_quant_test[index_nbr *
                                              LBLLENGTH:index_nbr * LBLLENGTH + LBLLENGTH]

                # compute dxdt
                delta_t = (time_nbr - time).item()
                dxhatstardt = (label_nbr - label) / delta_t

                # compute dgdt
                approximated_velocity = torch.matmul(
                    dgdx[i, :, :], dxhatstardt)
                approximated_velocity = approximated_velocity.view(-1, 3)

                # obtain ground truth
                u_nbr = snapshot_nbr_item['u']
                u_nbr = u_nbr.to(device)
                v_nbr = v_nbrs[i].float().to(
                                device)

                u_nbr = invstandardizeU(u_nbr.detach().cpu().numpy())
                u_nbr = torch.from_numpy(u_nbr)

                u_local = invstandardizeU(u_local.detach().cpu().numpy())
                u_local = torch.from_numpy(u_local)

                if use_actual_v:
                    actual_velocity = invstandardizeUScaleOnly(v_nbr.view(-1,v_nbr.size(2)).detach().cpu().numpy())
                    actual_velocity = torch.from_numpy(actual_velocity)
                else:
                    actual_velocity = (u_nbr - u_local) / delta_t
                    actual_velocity = actual_velocity[0,:,:]

                # invert to real velocity
                approximated_velocity = invstandardizeUScaleOnly(
                    approximated_velocity.detach().cpu().numpy())
                velocity_dict_pred[filename_nbr] = approximated_velocity.reshape(-1, approximated_velocity.shape[1])
                approximated_velocity = torch.from_numpy(approximated_velocity)

                zero = torch.zeros_like(approximated_velocity)
                zero = zero.to(approximated_velocity)

                val_loss = criterion_sum(
                    approximated_velocity, actual_velocity)
                val_norm = criterion_sum(zero, actual_velocity)
                running_vel_loss_sum_raw += val_loss
                running_vel_norm_sum_raw += val_norm

                basename = os.path.basename(filename)
                if target_dirname:
                    print('abs: ', val_loss)
                    print('rel: ', val_loss/val_norm)
                if basename == 'h5_f_0000000000.h5':
                    # print('approximated: ', approximated_velocity)
                    # print('      actual: ', actual_velocity)
                    # print('     u_local: ', u_local)
                    print('relative err: %.12f' % math.sqrt(val_loss / val_norm))
                    print(filename)
                    # print('label: ', label)
                    # print('label_nbr: ', label_nbr)
                    # print(dxhatstardt)

    outputs = net(lbl)
    if disp_type == 'u':
        error = criterion(outputs, u)
    elif disp_type == 'q':
        error = criterion(outputs, q)

    avg_error += error.item()

    zero = torch.zeros_like(q)
    zero = zero.to(q)
    if disp_type == 'u':
        error_sum = criterion_sum(outputs, u)
        squared_norm = criterion_sum(zero, u)
    elif disp_type == 'q':
        error_sum = criterion_sum(outputs, q)
        squared_norm = criterion_sum(zero, q)

    outputs = outputs.reshape_as(q_backup)
    u = u.reshape_as(q_backup)

    if preprocessing_type == 'stan':
        outputs = invstandardizeU(outputs.detach().cpu().numpy())
        u = invstandardizeU(u.detach().cpu().numpy())
    elif preprocessing_type == 'norm':
        outputs = invtransformU(outputs.detach().cpu().numpy())
        u = invtransformU(u.detach().cpu().numpy())
    outputs = torch.from_numpy(outputs)
    u = torch.from_numpy(u)
    q0_original_backup = q0_original_backup.view_as(outputs)

    outputs_q = outputs+q0_original_backup
    u_q = u+q0_original_backup

    for i in range(outputs_q.shape[0]):
        pos_output = outputs_q[i,:].reshape(-1, outputs_q.shape[3]).detach().cpu().numpy()
        position_write_list.append(pos_output)
        filename_output = batch['filename'][i]
        filename_write_list.append(filename_output)
        dy_dx_output = dy_dx_full[i,:].detach().cpu().numpy()
        deformation_write_list.append(dy_dx_output)

    if disp_type == 'u':
        zero = torch.zeros_like(u_q)
        zero = zero.to(u_q)
        loss_sum_raw = criterion_sum(outputs, u)
        loss_raw = criterion(outputs, u)
        squared_norm_raw = criterion_sum(zero, u)

        loss_sum_raw_q = criterion_sum(outputs_q, u_q)
        loss_raw_q = criterion(outputs_q, u_q)
        u_q_offset = u_q-position_offset
        squared_norm_raw_q = criterion_sum(zero, u_q_offset)
        
    elif disp_type == 'q':
        zero = torch.zeros_like(q)
        zero = zero.to(q)
        loss_sum_raw = criterion_sum(outputs, q)
        loss_raw = criterion(outputs, q)
        squared_norm_raw = criterion_sum(zero, q)

    running_error_sum += float(error_sum)
    running_norm_sum += float(squared_norm)

    running_loss_raw += float(loss_raw.item())
    running_loss_sum_raw += float(loss_sum_raw.item())
    running_norm_sum_raw += float(squared_norm_raw.item())

    running_loss_raw_q += float(loss_raw_q.item())
    running_loss_sum_raw_q += float(loss_sum_raw_q.item())
    running_norm_sum_raw_q += float(squared_norm_raw_q.item())

avg_error /= len(testloader)
pos_error = math.sqrt(running_loss_sum_raw_q / running_norm_sum_raw_q)
def_error = math.sqrt(running_def_loss_sum_raw / running_def_norm_sum_raw) 
vel_error = math.sqrt(running_vel_loss_sum_raw/running_vel_norm_sum_raw)
print('error pro dis: %.12f,  %.12f' %
      (avg_error / len(testloader), math.sqrt(running_error_sum/running_norm_sum)))
print('error raw dis: %.12f,  %.12f' %
      (running_loss_raw / len(testloader), math.sqrt(running_loss_sum_raw / running_norm_sum_raw)))
print('error raw pos: %.12f,  %.12f' %
      (running_loss_raw_q / len(testloader), pos_error))
print('error raw def: %.12f,  %.12f' %
      (0., def_error))
print('error raw F-I: %.12f,  %.12f' %
      (0., math.sqrt(running_def_loss_sum_raw_minusI / running_def_norm_sum_raw_minusI)))
if compute_generalized_velocity:
    print('error raw vel: %.12f,  %.12f' %
        (0., vel_error))
# print for excel
print('pos', 'vel', 'def')
print(pos_error, vel_error, def_error)

if recover:
    write_data(position_write_list, filename_write_list, deformation_write_list, velocity_dict_pred, velocity_dict_original, split_prefix, snap_type, label_write_list, filename_model)
    print('recover done.')

cov.stop()
cov.save()

cov.html_report(directory='test_covhtml')