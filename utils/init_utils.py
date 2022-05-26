import os
import os.path as osp
import random
import datetime
import itertools
import glob
import subprocess
import pyhocon
import glob
import re
import numpy as np
import glog as log

import torch

import sys
sys.path.append('../')

from models import joint_model, visual_dialog_encoder, coref_model

from dataloader.dataloader_visdial import VisdialDataset
from dataloader.dataloader_visdial_coref import VisdialPrpDataset
from dataloader.dataloader_visdial_coref_only import VisdialPrpOnlyDataset
from dataloader.dataloader_visdial_dense import VisdialDenseDataset


def load_runner(config):
    if config['model_type'] == 'conly':
        return coref_model.CorefRunner(config)
    if config['model_type'] == 'joint':
        if config['train_on_dense']:
            return joint_model.JointDenseRunner(config)
        else:
            return joint_model.JointRunner(config)
    elif config['model_type'] == 'vonly':
        if config['train_on_dense']:
            return visual_dialog_encoder.VisdialDenseRunner(config)
        else:
            return visual_dialog_encoder.VisdialRunner(config)


def load_dataset(config):
    dataset_eval = None

    if config['model_type'] == 'conly':
        dataset = VisdialPrpOnlyDataset(config)
    elif config['train_on_dense']:
        dataset = VisdialDenseDataset(config)
        if config['skip_mrr_eval']:
            temp = config['num_options_dense']
            config['num_options_dense'] = config['num_options']
            dataset_eval = VisdialDenseDataset(config)
            config['num_options_dense'] = temp
        else:
            dataset_eval = VisdialDataset(config)
    else:
        if config['model_type'] == 'joint':
            dataset = VisdialPrpDataset(config)
        elif config['model_type'] == 'vonly':
            dataset = VisdialDataset(config)
            if config['skip_mrr_eval']:
                dataset_eval = VisdialDenseDataset(config)

    if config['use_trainval']:
        dataset.split = 'trainval'
    else:
        dataset.split = 'train'
    if dataset_eval is not None:
        dataset_eval.split = 'val'
        
    return dataset, dataset_eval


def initialize_from_env(model, mode, model_type):
    if "GPU" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['GPU']

    config = pyhocon.ConfigFactory.parse_file(f"config/{model_type}.conf")[model]
    config['model_type'] = model_type
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        config['num_gpus'] = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        # multi-gpu setting
        if config['num_gpus'] > 1:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '5678'

    if mode == 'debug':
        model += '_debug'

    config["log_dir"] = os.path.join(config["log_dir"], model)
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    config['visdial_output_dir'] = osp.join(config['log_dir'], config['visdial_output_dir'])
    config['coref_output_dir'] = osp.join(config['log_dir'], config['coref_output_dir'])

    config['timestamp'] = datetime.datetime.now().strftime('%m%d-%H%M%S')

    if mode in ['predict', 'eval']:
        if (not config['loads_start_path']) and (not config['loads_best_ckpt']):
            config['loads_best_ckpt'] = True
            print(f'Setting loads_best_ckpt=True under predict or eval mode')
        if config['num_options_dense'] < 100:
            config['num_options_dense'] = 100
            print('Setting num_options_dense=100 under predict or eval mode')

    return config


def set_log_file(fname, file_only=False):
    # if fname already exists, find all log file under log dir,
    # and name the current log file with a new number
    if osp.exists(fname):
        prefix, suffix = osp.splitext(fname)
        log_files = glob.glob(prefix + '*' + suffix)
        count = 0
        for log_file in log_files:
            num = re.search(r'(\d+)', log_file)
            if num is not None:
                num = int(num.group(0))
                count = max(num, count)
        fname = fname.replace(suffix, str(count + 1) + suffix)
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        log.logger.handlers[0].stream = log.handler.stream = sys.stdout = sys.stderr = f = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def copy_file_to_log(log_dir):
    dirs_to_cp = ['.', 'config', 'coref', 'dataloader', 'models', 'preprocessing', 'utils']
    files_to_cp = ['*.py', '*.json', '*.sh', '*.conf']
    for dir_name in dirs_to_cp:
        dir_name = osp.join(log_dir, 'code', dir_name)
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
    for dir_name, file_name in itertools.product(dirs_to_cp, files_to_cp):
        filename = osp.join(dir_name, file_name)
        if len(glob.glob(filename)) > 0:
            os.system(f'cp {filename} {osp.join(log_dir, "code", dir_name)}')
    log.info(f'Files copied to {osp.join(log_dir, "code")}')


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def set_training_steps(config, num_samples):
    if config['parallel'] and config['dp_type'] == 'dp':
        config['num_iter_per_epoch'] = int(np.ceil(num_samples / config['batch_size']))
    else:
        config['num_iter_per_epoch'] = int(np.ceil(num_samples / (config['batch_size'] * config['num_gpus'])))
    if 'train_steps' not in config:
        config['train_steps'] = config['num_iter_per_epoch'] * config['num_epochs']
    if 'warmup_steps' not in config:
        config['warmup_steps'] = int(config['train_steps'] * config['warmup_ratio'])
    return config