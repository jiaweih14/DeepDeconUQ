"""
Part of the code is taken from https://github.com/Shai128/mqr
"""

import numpy as np
import torch as torch
from tqdm import tqdm
from model import MultivariateQuantileModel
from torch.utils.data import DataLoader, TensorDataset
from model import multivariate_qr_loss
import argparse
import matplotlib
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import warnings
import ast
from dataset import get_split_data
from helper import evaluate_performance, set_seeds
import logging

warnings.filterwarnings("ignore")

from sys import platform

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

def log_setting(log_dir, name='cquq', verbose=None):
    script_name = os.path.basename(__file__)
    script_name = script_name.rsplit('.', 1)[0]
    logging.captureWarnings(True)
    logger = logging.getLogger(__name__)

    # root log listens to everything
    root = logging.getLogger('')
    root.setLevel(logging.DEBUG)

    # log message format
    formatter = logging.Formatter(fmt='%(levelname)-8s | %(asctime)s | %(name)7s | %(message)s')

    # Runtime console listens to INFO by default
    ch = logging.StreamHandler()
    if verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File log listens to all levels from root
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_path = os.path.join(log_dir, script_name+'_'+name+'_'+now+'.log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Add some environmental details
    logger.debug(sys.version.replace('\n', ' '))
    return logger

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--tau', type=float, default=0.1,
                        help='quantile level')
    parser.add_argument('--suppress_plots', type=int, default=0,
                        help='1 to disable all plots, or 0 to allow plots')

    parser.add_argument('--dataset_name', type=str, default='banana',
                        help='dataset to use')

    parser.add_argument('--num_q', type=int, default=32,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
                        help='number of epochs')

    parser.add_argument('--hs', type=str, default="[64, 64, 64]",
                        help='hidden dimensions')

    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=256,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=100,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--ds_type', type=str, default="REAL",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='ratio of test set size')
    parser.add_argument('--calibration_ratio', type=float, default=0.4,  # 0.5 of training size
                        help='ratio of calibration set size')

    parser.add_argument('--save_training_results', type=int, default=0,
                        help='1 for saving results during training, or 0 for not saving')
    parser.add_argument('--transform', type=str, default="identity",
                        help='')
    parser.add_argument('--ct', type=str, default="['malignant']",
                        help="celltypes when applying onto aml or pbmc data")
    parser.add_argument('--vae_loss', type=str, default="KL",
                        help="'KL' or 'MMD'")
    parser.add_argument('--vae_z_dim', type=int, default=3,
                        help="encoded dimension")
    parser.add_argument('--out_dir', type=str, default='./NVQR_results/',
                        help="model and prediction output directory")
    parser.add_argument('--pca', type=int, default= 1,
                        help="perform pca on the data (1) or not (0)")
    parser.add_argument('--noise', type=int, default= 1,
                        help="select noise data (1, 5, or 10)")
    parser.add_argument('--softmax', type=int, default= 0,
                        help="add softmax activation to the output layer (1) or not (0)")
    parser.add_argument('--cali_factor', type=float, default= 0.0,
                        help="calibration factor for the calibration")
    parser.add_argument('--scaler', type=str, default= 'standard',
                        help="standard or minmax scaler")
    parser.add_argument('--test_dataset', type=str, default= 'None',
                        help="testing dataset name, None means no testing dataset, use test_ratio to split the data")
    args = parser.parse_args()

    assert 'identity' in args.transform

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device
    args.num_ens = 1
    args.boot = 0
    args.hs = ast.literal_eval(args.hs)
    args.ct = ast.literal_eval(args.ct)
    args.test_dataset = ast.literal_eval(args.test_dataset)
    args.suppress_plots = False if args.suppress_plots == 0 else 1
    
    return args


if __name__ == '__main__':
    loss = 'pinball'
    TRAINING_OVER_ALL_QUANTILES = 'int' in loss
    args = parse_args()
    print(f"out dir {args.out_dir}")
    logger = log_setting(args.out_dir, name = args.dataset_name)
    logger.info("DEVICE: {}".format(args.device))
    logger.debug(f'Command line: {args}')
    dataset_name = args.dataset_name
    device = args.device

    seed = args.seed
    set_seeds(seed)

    test_ratio = args.test_ratio
    calibration_ratio = args.calibration_ratio
    val_ratio = 0.1
 
    dan = 'dan' in args.ds_type.lower()
    scale = 'real' in args.ds_type.lower() and 'dan' not in args.ds_type.lower()
    data = get_split_data(dataset_name, device, test_ratio, val_ratio, calibration_ratio, seed, scale, \
                                   scaler = args.scaler, pca=args.pca, testing_dataset=args.test_dataset, dan = dan, noise=args.noise)
    if data['cts'] is not None:
        args.cts = data['cts']
    x_train, x_val, y_train, y_val, x_test, y_te, = data['x_train'], data['x_val'], \
                                                    data['y_train'], data['y_val'], \
                                                    data['x_test'], data['y_te']
    print(f"train set size: x_train: {x_train.shape}, y_train: {y_train.shape}, contain nan: {np.isnan(x_train).any()}")
    print(f"validation set size: x_val: {x_val.shape}, y_val: {y_val.shape}, contain nan: {np.isnan(x_val).any()}")
    print(f"test set size: x_test: {x_test.shape}, y_te: {y_te.shape}, contain nan: {np.isnan(x_test).any()}")
    np.save(os.path.join(args.out_dir, 'test_x.npy'), x_test.cpu().numpy())
    np.save(os.path.join(args.out_dir, 'test_y.npy'), y_te.cpu().numpy())
    scale_x = data['scale_x']
    scale_y = data['scale_y']
    x_dim = x_train.shape[1]

    # if dataset_name in ['aml_primary', 'aml_recurrent', 'aml_beat']:
    d = y_train.shape[1] if args.cali_factor == 0 else args.cali_factor
    tau_per_dimension = args.tau / d  # beta = 1 - alpha/d

    args.conformalization_tau = args.tau
    args.tau_list = torch.Tensor([args.tau]).to(device)
    # else:
    #     d = 2
    #     tau_per_dimension = args.tau / d  # beta = 1 - alpha/d
    #     args.conformalization_tau = args.tau  # total desired coverage
    #     args.tau = tau_per_dimension
    #     args.tau_list = torch.Tensor([args.tau]).to(device)


    print(f"dataset_name: {dataset_name}, tau: {args.tau}, conformalization tau: {args.conformalization_tau}, seed: {seed}")
    if calibration_ratio > 0:
        x_cal, y_cal = data['x_cal'], data['y_cal']
        np.save(os.path.join(args.out_dir, 'calibration_x.npy'), x_cal.cpu().numpy())
        np.save(os.path.join(args.out_dir, 'calibration_y.npy'), y_cal.cpu().numpy())
        logger.info(f"calibration set size: x_cal: {x_cal.shape}, y_cal: {y_cal.shape}")

    dim_y = y_train.shape[1]
    model = MultivariateQuantileModel(input_size=x_dim, nn_input_size=x_dim + 1, output_size=dim_y, y_size=dim_y,
                                           hidden_dimensions=args.hs, dropout=args.dropout,
                                           lr=args.lr, wd=args.wd, num_ens=args.num_ens, device=args.device,
                                           softmax_output=args.softmax, args =args)

    loader = DataLoader(TensorDataset(x_train, y_train),
                        shuffle=True,
                        batch_size=args.bs)

    # Loss function
    loss_fn = multivariate_qr_loss
    batch_loss = True
    args.tau_list = torch.Tensor([args.tau]).to(device)
    alpha = args.tau
    assert len(args.tau_list) == 1
    va_loss_list = []
    tr_loss_list = []

    for ep in tqdm(range(args.num_ep)):

        if model.done_training:
            break

        # Take train step
        ep_train_loss = []  # list of losses from each batch, for one epoch
        for (xi, yi) in loader:
            if TRAINING_OVER_ALL_QUANTILES:
                q_list = torch.rand(args.num_q)
            else:
                q_list = torch.Tensor([alpha / 2])
            loss = model.loss(loss_fn, xi, yi, q_list,
                              batch_q=batch_loss,
                              take_step=True, args=args)
            ep_train_loss.append(loss)

        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0)
        tr_loss_list.append(ep_tr_loss)

        # Validation loss
        if TRAINING_OVER_ALL_QUANTILES:
            va_te_q_list = torch.linspace(0.01, 0.99, 99)
        else:
            va_te_q_list = torch.Tensor([alpha / 2, 1 - alpha / 2])
        ep_va_loss = model.update_va_loss(
            loss_fn, x_val, y_val, va_te_q_list,
            batch_q=batch_loss, curr_ep=ep, num_wait=args.wait,
            args=args
        )
        va_loss_list.append(ep_va_loss)

    params = {'dataset_name': dataset_name, 'epoch': model.best_va_ep[0],
              'seed': seed, 'tau': args.conformalization_tau,
              'dropout': args.dropout, 'hs': str(args.hs)}
    logger.info(f'params: {params}')

    if args.calibration_ratio > 0:
        scores = model.conformalize(x_cal, y_cal, args.conformalization_tau, args.tau)
        np.save(os.path.join(args.out_dir, 'calibration_scores.npy'), scores.cpu().numpy())
        torch.save(model, os.path.join(args.out_dir, 'model.pth'))
        results = evaluate_performance(model, dataset_name, x_test, y_te,args=args)
        