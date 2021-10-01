import os
import argparse

parser = argparse.ArgumentParser(
    description='Rename files')
parser.add_argument('-d', help='path to the dataset',
                    type=str, nargs=1, required=True)
parser.add_argument('--dry_run', dest='dry_run', action='store_true')
args = parser.parse_args()    

dry_run = args.dry_run
data_path = args.d[0]
if data_path.endswith('/'):
    data_path = data_path[:-1]
dir_list = os.listdir(data_path)
dir_list_true = []
for dirname in dir_list:
    print(dirname)
    if os.path.isdir(os.path.join(data_path,dirname)):
        dir_list_true.append(dirname)
for dirname in dir_list_true:
    old_path_name = os.path.join(data_path,dirname)
    new_path_name = os.path.join(data_path,'sim_seq_' + dirname)
    mv_command = 'mv ' + old_path_name + ' ' + new_path_name
    print(mv_command)
    print()
    if not dry_run:
        os.system(mv_command)