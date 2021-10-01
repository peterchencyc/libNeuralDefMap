from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PrepareDataset import *
from timeit import default_timer as timer
import random
from datetime import datetime
import argparse
import time
from torch.optim.lr_scheduler import LambdaLR
import cProfile
import git
import coverage

cov = coverage.Coverage()
cov.start()

parser = argparse.ArgumentParser(
    description='ROM training')
parser.add_argument('-d', help='path to the dataset',
                    type=str, nargs=1, required=True)
parser.add_argument('-m', help='path to the saved weights',
                    type=str, nargs=1, required=False)
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
parser.add_argument('-alr', help='adaptive learning rate',
                    dest='alr', action='store_true')
parser.add_argument('-lr', help='learning rate',
                    type=float, nargs='*', required=True)
parser.add_argument('-epo', help='epoch size',
                    type=int, nargs='*', required=True)
parser.add_argument('-lbl', help='label length',
                    type=int, nargs=1, required=True)
parser.add_argument('-cdm', help='deformation gradient penalty',
                    type=float, nargs=1, required=False)
parser.add_argument('-train_ratio', help='train / test splitting',
                    type=float, nargs=1, required=True)
parser.add_argument('-cgv', help='velocity penalty',
                    type=float, nargs=1, required=False)
parser.add_argument('-debug', dest='debug',
                    action='store_true')
parser.add_argument('-use_special_label', dest='use_special_label',
                    action='store_true')
parser.add_argument('-use_actual_v', dest='use_actual_v',
                    action='store_true')
parser.add_argument('-log', help='folder to store log',
                    type=str, nargs=1, required=False)
parser.add_argument('-optimizer', type=str, nargs=1, required=False)
parser.add_argument('-print_every', help='print_every',
                    type=int, nargs=1, required=False)
parser.add_argument('-save_every', help='print_every',
                    type=int, nargs=1, required=False)
parser.add_argument('-time_string', 
                    type=str, required=False)
parser.add_argument('-use_inverse_norm', dest='use_inverse_norm',
                    action='store_true')
parser.add_argument('-ini_label_type', type=str, nargs=1, required=False)
parser.add_argument('-samp_start_number', help='samp_start_number',
                    type=int, nargs=1, required=False)
parser.add_argument('-samp_gap', help='samp_gap',
                    type=int, nargs=1, required=False)
parser.add_argument('-temporal_sampling_every_n', help='temporal_sampling_every_n',
                    type=int, nargs=1, required=False)
args = parser.parse_args()

EPOCH_SIZE = args.epo[0]
data_path = args.d[0]
if data_path.endswith('/'):
    data_path = data_path[:-1]
use_actual_v = True
use_inverse_norm = True
use_special_label = True
sup_type = 'full'
    

LBLLENGTH = args.lbl[0]
samp_start_number = args.samp_start_number[0] if args.samp_start_number else 0
samp_gap = args.samp_gap[0] if args.samp_gap else 1
temporal_sampling_every_n = args.temporal_sampling_every_n[0] if args.temporal_sampling_every_n else 1

if len(args.lr)>1:
    adaptiveLRfromRange, EPOCH_SIZE = generateLR(args.lr, args.epo)
    learning_rate = 1e-3
else:
    learning_rate = args.lr[0]
    adaptiveLRfromRange = None

inter = None

if args.cgv:
    lambda_mult_cgv = args.cgv[0]
    if lambda_mult_cgv>0:
        compute_generalized_velocity = True
    else:
        compute_generalized_velocity = False
else:
    compute_generalized_velocity = False
    lambda_mult_cgv = 0

if args.cdm:
    lambda_mult = args.cdm[0]
    if lambda_mult>0:
        compare_def_map = True
    else:
        compare_def_map = False
else:
    compare_def_map = False
    lambda_mult = 0

if args.optimizer:
    optimizer_option = args.optimizer[0]
else:
    optimizer_option = 'adam'

onlyinifilename = None

ini_label_type = 'none'

print_every = 20
if args.print_every:
    print_every = args.print_every[0]
save_every = 100
if args.save_every:
    save_every = args.save_every[0]

if args.time_string:
    time_string = args.time_string
else:
    time_string = time.strftime("%Y%m%d-%H%M%S")

filename_manager = FilenameManager()
filename_manager.increment = temporal_sampling_every_n

if args.log:
    logname = args.log[0]
    logname = os.path.join(logname, 'log_' + time_string + '_' + os.path.basename(data_path))
    logname += '_sup-' + sup_type
    logname += '_epo-' + str(EPOCH_SIZE)
    logname += '_opt-' + optimizer_option
    if compute_generalized_velocity:
        logname += '_cgv-' + str(lambda_mult_cgv)
    if compare_def_map:
        logname += '_cdm-' + str(lambda_mult)
    logname += '_lbl-' + str(LBLLENGTH)
    if use_actual_v:
        logname += '_use_actual_v'
    if use_special_label:
        logname += '_use_special_label'
    logname += '.txt'
    print("print to log: ", logname)
    logfile = open(logname, 'w')
    sys.stdout = logfile
    sys.stderr = logfile

print("input data path: ", data_path)
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print('GIT hash: ', sha)

snap_type = 'automapconv'
disp_type = 'u'
preprocessing_type = 'stan'


time_only = False
create_graph = True
use_real_time = False
DEBUG = args.debug
def log(*argv):
    if DEBUG:
        print(*argv)

# use 1.0 if want to use all the data for training
train_ratio = args.train_ratio[0]
if train_ratio > 1.0 or train_ratio < 0.0:
    exit('invalid train_ratio')

if args.np:
    number_points = args.np[0]
else:
    number_points = -1

if number_points == -1:
    point_indices = None
else:
    point_indices = list(range(0, number_points))
samp_gap_enc = samp_gap
full_dataset, train_dataset, _, preprop, point_indices, point_indices_enc = prepareData(
    data_path, point_indices, preprocessing_type, train_ratio, use_real_time, LBLLENGTH, onlyinifilename, ini_label_type, samp_start_number, samp_gap, samp_gap_enc, temporal_sampling_every_n)
device, net, criterion, BATCH_SIZE, _ = prepareOther(
    snap_type, time_only, point_indices_enc, LBLLENGTH)
criterion_sum = nn.MSELoss(reduction='sum')
invtransformU = preprop.computeInvNormalizationTransformationU()
invstandardizeU = preprop.computeInvStandardizeU()
invstandardizeQ0 = preprop.computeInvStandardizeQ0()
invstandardizeUTorch = preprop.computeInvStandardizeUTorch(device)
transform_output_torch = preprop.computeNormalizationTransformationTorch(device)

if args.m:
    PATH = args.m[0]

train_ratio_dir = 'train_ratio-' + str(train_ratio)
train_ratio_dir_full = os.path.join(data_path, train_ratio_dir)
os.makedirs(train_ratio_dir_full, exist_ok=True)
error_type = 'train'
dataset_filenames = obtainFilenames(data_path, train_dataset, train_ratio_dir, error_type, filename_manager)

if snap_type == 'automapconv':
    net_enc = NetAutoEncEnc(net)
    net_dec = NetAutoDec(net)
    net_dec_grad = NetGrad(net)
    hidden_quant=None
else:
    exit('invalid snap_type')

if sup_type == 'full':
    opt_quant = net.parameters()
else:
    exit('invalid sup_type')

if compute_generalized_velocity or inter:
    filename_index_map = buildFilenameIndexMap(dataset_filenames)
    assignNbrId(train_dataset, filename_index_map, dataset_filenames, filename_manager)

trainloader = FastDataLoader(train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=64)

def init_normal(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        random.seed(datetime.now())
        seed_number = random.randint(0, 100)
        random.seed(0)
        torch.manual_seed(seed_number)
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        elif type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
        torch.manual_seed(torch.initial_seed())


net.apply(init_normal)

# for some reason, init normal mess up initial_seed so we need to re-seed again
torch.manual_seed(0)
# x = 0
# for param in net.parameters():
#     print('parm shape: ', param.shape)
#     x += 1
# if snap_type == 'map':
#     param.detach()[:] = torch.zeros_like(param)
#     if x == 4:
#         param.detach()[:] = torch.tensor([0.5049, 0.0000, 0.4979])
#         print(param)

if args.m:
    net.load_state_dict(torch.load(PATH))

if optimizer_option == 'adam':
    optimizer = optim.Adam(opt_quant, lr=learning_rate)
elif optimizer_option == 'lbfgs':
    optimizer = optim.LBFGS(opt_quant, lr=learning_rate,
                        line_search_fn='strong_wolfe')
else:
    exit('unsupported optimizer')

if adaptiveLRfromRange:
    scheduler = LambdaLR(optimizer, lr_lambda=adaptiveLRfromRange)
else:
    if args.alr:
        if snap_type == 'automapconv':
            scheduler = LambdaLR(optimizer, lr_lambda=adaptiveLRConv)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=regularLR)


def count_parameters(optimizer):
    return sum(parm.numel() for parm in optimizer.param_groups[0]['params'])


print('Number of network parameters: ', count_parameters(optimizer))

output_path = './autoencoder-'+snap_type + \
        '_'+time_string
def saveModel():
    torch.save(net.state_dict(), output_path + '.pth')
    print('writing: ', output_path + '.pth')
    if hidden_quant is not None:
        torch.save(hidden_quant, output_path+'.hid')

initial_time = timer()
# net = nn.DataParallel(net)

if DEBUG:
    pr = cProfile.Profile()
    pr.enable()
try:
    for epoch in range(EPOCH_SIZE):  # loop over the dataset multiple times
        epoch_time = timer()
        running_loss = 0.0
        running_loss_sum = 0.0
        running_norm_sum = 0.0
        running_loss_raw = 0.0
        running_loss_sum_raw = 0.0
        running_norm_sum_raw = 0.0
        running_def_loss_sum_raw = 0.0
        running_def_norm_sum_raw = 0.0
        running_vel_loss_sum_raw = 0.0
        running_vel_norm_sum_raw = 0.0
        batch_time_sum = 0
        do_print_epoch = (epoch % print_every == print_every - 1)
        for i_batch, data_batched in enumerate(trainloader):
            do_print_batch = (i_batch % print_every == print_every - 1)
            batch_time = timer()
            q_backup = data_batched['q'].float().detach()
            def closure():
                start_closure = timer()
                start = timer()
                global running_def_loss_sum_raw
                global running_def_norm_sum_raw
                global running_vel_loss_sum_raw
                global running_vel_norm_sum_raw
                q, u, lbl, v, lbl_mid_list, u_mid_list = dataFromBatch(
                    data_path, data_batched, snap_type, sup_type, hidden_quant, LBLLENGTH, device, time_only, inter)

                v_local = v.to(device)                            
                u_local = u.to(device)
                lbl_local = lbl.to(device)

                log(' prep: ', timer() - start)

                start = timer()
                optimizer.zero_grad()

                if compute_generalized_velocity and sup_type == 'full':
                    lbl_local = lbl_local.detach()
                    lbl_local = lbl_local.to(device)
                    lbl_local.requires_grad_(True)

                outputs_local = net(lbl_local)

                

                if disp_type == 'u':
                    if use_inverse_norm:
                        loss = 1000 * criterion(invstandardizeUTorch(outputs_local), invstandardizeUTorch(u_local))
                    else:
                        loss = criterion(outputs_local, u_local)
                    if u_mid_list is not None:
                        exit('invalid u_mid_list')

                else:
                    exit('invalid disp_type')
                log(' net: ', timer() - start)
                
                start = timer()
                if compare_def_map:
                    if snap_type == 'automapconv':
                        us = data_batched['u'].float()
                        batch_size_local = us.size(0)
                        npoints = us.size(2)

                        us = us.view(us.size(0),us.size(2),us.size(3))
                        us = us.to(device)
                        xhats = net_enc(us)
                        lbl_local = prepNetworkInputFromXhat(device, data_batched, xhats)

                        dy_dx_full = dydxFromNetGrad(lbl_local, preprop, net_dec_grad, device).view(batch_size_local, npoints, 9)

                        f_tensor = defmapFromBatch(data_batched, device)
                        loss += lambda_mult * criterion(dy_dx_full, f_tensor)

                        if do_print_batch or do_print_epoch:
                            # compute loss information for viewing
                            dy_dx_full = dy_dx_full.detach()
                            f_tensor = f_tensor.detach()
                            zero = torch.zeros_like(f_tensor)
                            zero = zero.to(f_tensor)
                            running_def_loss_sum_raw += criterion_sum(
                                dy_dx_full, f_tensor)
                            running_def_norm_sum_raw += criterion_sum(
                                zero, f_tensor)
                    else:
                        exit('invalid snap_type')

                log(' cdm: ', timer() - start)
                start = timer()
                if compute_generalized_velocity:
                    if snap_type == 'automapconv':
                        loss_cgv = torch.zeros(1).to(device)
                        start_cgv = timer()
                        us = data_batched['u'].float()
                        u_nbrs = data_batched['u_nbr'].float()
                        us = us.view(us.size(0),us.size(2),us.size(3))
                        u_nbrs = u_nbrs.view(u_nbrs.size(0),u_nbrs.size(2),u_nbrs.size(3))
                        us = us.to(device)
                        u_nbrs = u_nbrs.to(device)
                        u_alls = torch.cat((us, u_nbrs),0)
                        xhat_alls = net_enc(u_alls)

                        batch_size_local = us.size(0)
                        xhats = xhat_alls[:batch_size_local,:,:]
                        xhat_nbrs = xhat_alls[-batch_size_local:,:,:]

                        lbl_local = prepNetworkInputFromXhat(device, data_batched, xhats)

                        outputs_grad = net_dec_grad(lbl_local)
                        dgdx_analy = outputs_grad[:, :, :LBLLENGTH]
                        dgdx_analy = dgdx_analy.view(batch_size_local, -1, dgdx_analy.size(2))
                        dgdx = dgdx_analy
                        log('   cgv_net: ', timer() - start_cgv)

                        u_batch = u.view(batch_size_local, -1, u.size(1))
                        v_batch = v.view(batch_size_local, -1, v.size(1))

                        # compute dgdt thru dxdt
                        filenames = data_batched['filename']
                        times = data_batched['time']
                        id_cons = data_batched['id_con']
                        id_nbrs = data_batched['id_nbr']
                        time_nbrs = data_batched['time_nbr']
                        v_nbrs = data_batched['v_nbr']

                        for i in range(len(id_cons)):
                            filename = filenames[i]
                            time = times[i]
                            label = xhats[i,0,:]
                            index_nbr = id_nbrs[i]

                            if index_nbr > -1:
                                time_nbr = time_nbrs[i].item()
                                label_nbr = xhat_nbrs[i,0,:]

                                # compute dxdt
                                delta_t = (time_nbr - time).item()
                                dxhatstardt = (label_nbr - label) / delta_t

                                # compute dgdt
                                approximated_velocity = torch.matmul(
                                    dgdx[i, :, :], dxhatstardt)
                                approximated_velocity = approximated_velocity.view(-1, 3)

                                # obtain ground truth
                                u_nbr = u_nbrs[i].float().to(
                                    device)
                                v_nbr = v_nbrs[i].float().to(
                                    device)
                                if use_actual_v:
                                    actual_velocity = v_nbr.view(-1,v_nbr.size(2))
                                else:
                                    actual_velocity = (u_nbr - u_local) / delta_t
                                    actual_velocity = actual_velocity[0, :, :]
                                
                                if use_inverse_norm:
                                    loss += lambda_mult_cgv * 1000 * criterion(invstandardizeUTorch(approximated_velocity), invstandardizeUTorch(actual_velocity))
                                    
                                else:
                                    loss += lambda_mult_cgv * criterion(approximated_velocity,
                                                actual_velocity)

                                if do_print_batch or do_print_epoch:
                                    # invert to real velocity
                                    approximated_velocity = invstandardizeU(
                                        approximated_velocity.detach().cpu().numpy())
                                    approximated_velocity = torch.from_numpy(
                                        approximated_velocity)
                                    actual_velocity = invstandardizeU(
                                        actual_velocity.detach().cpu().numpy())
                                    actual_velocity = torch.from_numpy(
                                        actual_velocity)

                                    zero = torch.zeros_like(approximated_velocity)
                                    zero = zero.to(approximated_velocity)

                                    val_loss = criterion_sum(
                                        approximated_velocity, actual_velocity)
                                    val_norm = criterion_sum(zero, actual_velocity)
                                    running_vel_loss_sum_raw += val_loss
                                    running_vel_norm_sum_raw += val_norm
                    else:
                        exit('invalid snap_type')
                log(' cgv: ', timer() - start)
                start = timer()
                loss.backward(retain_graph=False)
                log(' descent: ', timer() - start)
                log(' closure: ', timer() - start_closure)
                return loss
            start = timer()
            loss = optimizer.step(closure)
            log('SGD: ', timer() - start)

            if do_print_batch or do_print_epoch:
                start = timer()
                q, u, lbl, _, _, _ = dataFromBatch(
                    data_path, data_batched, snap_type, sup_type, hidden_quant, LBLLENGTH, device, time_only)

                q = q.to(device)
                u = u.to(device)
                lbl = lbl.to(device)

                # compute relative error
                outputs = net(lbl)
                zero = torch.zeros_like(q)
                zero = zero.to(q)
                if disp_type == 'u':
                    loss_sum = criterion_sum(outputs, u)
                    loss = criterion(outputs, u)
                    squared_norm = criterion_sum(zero, u)
                else:
                    exit('invalid disp_type')
                outputs = outputs.reshape_as(q_backup)
                u = u.reshape_as(q_backup)

                if preprocessing_type == 'stan':
                    outputs = invstandardizeU(outputs.detach().cpu().numpy())
                    u = invstandardizeU(u.detach().cpu().numpy())
                else:
                    exit('invalid preprocessing_type')
                outputs = torch.from_numpy(outputs)
                u = torch.from_numpy(u)

                if disp_type == 'u':
                    zero = torch.zeros_like(u)
                    zero = zero.to(u)
                    loss_sum_raw = criterion_sum(outputs, u)
                    loss_raw = criterion(outputs, u)
                    squared_norm_raw = criterion_sum(zero, u)
                else:
                    exit('invalid disp_type')

                # print statistics
                running_loss += loss.item()
                running_loss_sum += loss_sum.item()
                running_norm_sum += squared_norm.item()
                running_loss_raw += loss_raw.item()
                running_loss_sum_raw += loss_sum_raw.item()
                running_norm_sum_raw += squared_norm_raw.item()
                
                log('stats: ', timer() - start)

            
                print('[%d, %5d] pro: %.12f,  %.12f' %
                      (epoch + 1, i_batch + 1, running_loss, math.sqrt(running_loss_sum / running_norm_sum)))
                print('[%d, %5d] raw: %.12f,  %.12f' %
                      (epoch + 1, i_batch + 1, running_loss_raw, math.sqrt(running_loss_sum_raw / running_norm_sum_raw)))
                if compare_def_map:
                    print('[%d, %5d] def: %.12f,  %.12f' %
                          (epoch + 1, i_batch + 1, 0.0, math.sqrt(running_def_loss_sum_raw / running_def_norm_sum_raw)))
                if compute_generalized_velocity and running_vel_norm_sum_raw > 0: # need the >0 for now due to the possibility of no pairing (todo: switch to Kevin's suggestion of implicit v hat)
                    print('[%d, %5d] vel: %.12f,  %.12f' %
                          (epoch + 1, i_batch + 1, 0.0, math.sqrt(running_vel_loss_sum_raw / running_vel_norm_sum_raw)))
                if math.isnan(running_loss):
                    exit('encounter nan; exit')
                running_loss = 0.0
                running_loss_raw = 0.0
            batch_time = timer() - batch_time
            log('batch_time: ', batch_time)
            batch_time_sum += batch_time
        scheduler.step()
        if (epoch + 1) % save_every == 0:
            saveModel()
        log('epoch: ', timer() - epoch_time)
        log('batch_time_sum: ', batch_time_sum)
    print('Finished Training')
    print('time used: ', timer() - initial_time)
    if DEBUG:
        pr.disable()
        pr.print_stats(sort='time')
    if not DEBUG:
        saveModel()
except KeyboardInterrupt:
    print('Finish earlier')
    print('time used: ', timer() - initial_time)
    # if not DEBUG:
    #     saveModel()

cov.stop()
cov.save()

cov.html_report()