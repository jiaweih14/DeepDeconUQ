"""
Part of the code is taken from https://github.com/Shai128/mqr
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import sys
import helper
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
import anndata


class Identity:
    def transform(self, y):
        return y
    def fit_transform(self, y):
        return y
class Normalizer:
    def __init__(self, log_transform=False, scaler='standard'):
        self.log_transform = log_transform
        self.scaler = scaler

    def fit(self, x):
        if self.log_transform:
            x = np.log(x + 1)
        if self.scaler == 'standard':
            self.normlizaer = StandardScaler()#.fit(x)
        elif self.scaler =='minmax':
            self.normlizaer = MinMaxScaler()#.fit(x)
        elif self.scaler == 'none':
            self.normlizaer = Identity()
        else:
            raise ValueError(f"Scaler {self.scaler} is not implemented")
        return self       

    def transform(self, x):
        if self.log_transform:
            x = np.log(x + 1)
        # return self.normlizaer.transform(x)
        return self.normlizaer.fit_transform(x.T).T


def data_train_test_split(Y, X=None, device='cpu', test_ratio=0.2, val_ratio=0.2,
                          calibration_ratio=0., seed=0, scale=False, dim_to_reduce=None, 
                          is_real=True, x_log_transform=False, x_scaler='standard', 
                          y_log_transform=False, y_scaler='standard', tail_test=0, pca=True):
    data = {}
    is_conditional = X is not None
    if X is not None:
        X = X.cpu()
    Y = Y.cpu()

    y_names = ['y_train', 'y_val', 'y_te']
    if is_conditional:
        x_names = ['x_train', 'x_val', 'x_test']
        if tail_test:
            x_train, x_test, y_train, y_te = X[:-tail_test, :], X[-tail_test:, :], Y[:-tail_test, :], Y[-tail_test:, :]
        else:
            x_train, x_test, y_train, y_te = train_test_split(X, Y, test_size=100, random_state=seed) #  test_ratio
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=seed)#

        if calibration_ratio > 0:
            x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=calibration_ratio,
                                                              random_state=seed) 
            # idx = np.random.choice(len(y_te), len(y_te)//2, replace=False)
            data['x_cal'] = x_cal #torch.cat((x_cal, x_test), dim=0)
            data['y_cal'] = y_cal #torch.cat((y_cal, y_te), dim=0)
            x_names += ['x_cal']
            y_names += ['y_cal']

        data['x_train'] = x_train
        data['x_val'] = x_val
        data['x_test'] = x_test

        if scale:
            s_tr_x = Normalizer(log_transform=x_log_transform, scaler=x_scaler).fit(x_train)
            data['s_tr_x'] = s_tr_x
            for x in x_names:
                data[x] = torch.Tensor(s_tr_x.transform(data[x]))

        if pca and ((is_real and x_train.shape[1] > 70) or (dim_to_reduce is not None and x_train.shape[1] > dim_to_reduce)):
            if dim_to_reduce is None:
                n_components = 50 if x_train.shape[1] < 150 else 600 # 600 is good
            else:
                n_components = dim_to_reduce
            pca = PCA(n_components=n_components)
            pca.fit(data['x_train'])
            for x in x_names:
                data[x] = torch.Tensor(pca.transform(data[x].numpy()))

        for x in x_names:
            data[x] = data[x].to(device)

    else:
        y_train, y_te = train_test_split(Y, test_size=test_ratio, random_state=seed)
        y_train, y_val = train_test_split(y_train, test_size=val_ratio, random_state=seed)
        if calibration_ratio > 0:
            y_train, y_cal = train_test_split(y_train, test_size=calibration_ratio, random_state=seed)
            y_names += ['y_cal']
            data['y_cal'] = y_cal

    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_te'] = y_te

    if scale:
        s_tr_y = Normalizer(log_transform=y_log_transform, scaler=y_scaler).fit(y_train)
        data['s_tr_y'] = s_tr_y

        for y in y_names:
            data[y] = torch.Tensor(s_tr_y.transform(data[y]))

    for y in y_names:
        data[y] = data[y].to(device)

    return data


def GetDataset(name, base_path, is_1d = False, testing_dataset=None, scale=True, noise=0):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""
    test_size = 0
    cts = None

    if name.startswith('aml'):
        subject = name[4:]
        if scale:
            adata = anndata.read_h5ad(base_path+'aml_data_no_beat.h5ad')
            prefix = subject+'_200'
            tmp = anndata.read_h5ad(base_path+prefix+'.h5ad')

        else:
            adata = anndata.read_h5ad(base_path+'dan/encoded_train1.h5ad')
            prefix = 'encoded_'+subject
            tmp = anndata.read_h5ad(base_path+'aml/data/dan/'+prefix+'.h5ad')
    
        adata = anndata.concat([adata, tmp], axis=0, uns_merge='same')
        if subject in ['primary', 'recurrent', 'beat']:
            test_size = int(0.9 * len(tmp))
        else:
            test_size = len(tmp)
        X = adata.X
        cts = 'malignant' if is_1d else adata.uns['cell_types']
        y = adata.obs[cts].values
        
    if name.startswith('pbmc'):
        adata = anndata.read_h5ad(base_path+'pbmc_data.h5ad')
        subject = name[5:] # for example pbmc_data8k
        if 'sdy67' in subject or 'GSE65133' in subject:
            tmp = anndata.read_h5ad(base_path+subject+'.h5ad')
            common = tmp.var.index.intersection(adata.var.index)
            adata = anndata.concat([adata[:, common], tmp[:, common]], axis=0, uns_merge='same')
            test_size = len(tmp)
        else:
            tmp1 = adata[adata.obs['ds'] == subject]
            tmp2 = adata[adata.obs['ds'] != subject]
            adata = anndata.concat([tmp2, tmp1], axis=0, uns_merge='same')
            test_size = 0
        X = adata.X
        cts = adata.uns['cell_types']
        y = adata.obs[cts].values
    
    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    except Exception as e:
        raise Exception("invalid dataset")

    return X, y, test_size, cts


def get_real_data(dataset_name, testing_dataset, scale=True, noise = 0):
    is_1d = '1d_' in dataset_name
    dataset_name = dataset_name.replace("1d_", "")
    X, y, test_size, cts = GetDataset(dataset_name, './UQ/aml/data/', is_1d, testing_dataset=testing_dataset, scale=scale, noise = noise)
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    Y = y
    if is_1d:
        Y = y.reshape(len(y), 1)
    return Y, X, test_size, cts


def get_dataset(dataset_name, testing_dataset, scale = True, noise=0):
    Y, X, test_size, cts = get_real_data(dataset_name, testing_dataset=testing_dataset, scale=scale, noise=noise)
    return Y, X, test_size, cts


def get_split_data(dataset_name, device, test_ratio, val_ratio, calibration_ratio, seed, scale, scaler='standard', pca=True, testing_dataset=None, dan=False, noise=0):
    dim_to_reduce = 10 if 'reduced_' in dataset_name else None
    dataset_name = dataset_name.replace("reduced_", "")
    x_log_transform =('aml' in dataset_name.lower() or 'pbmc' in dataset_name.lower()) and not dan 
    y_scaler = 'none' if 'aml' in dataset_name.lower() or 'pbmc' in dataset_name.lower() else scaler

    Y, X, test_size, cts = get_dataset(dataset_name, testing_dataset=testing_dataset, scale = scale, noise=noise)
    print(f'Y shape: {Y.shape}, X shape: {X.shape}, x log transform: {x_log_transform}, y scaler: {y_scaler}')
    print(f'Y: {Y[:5]}')
    print(f"x contain nan: {torch.isnan(X).any()}")
    print(f"scale: {scale}, pca: {pca}, dan: {dan}")
    data = data_train_test_split(Y, X=X, device=device,
                                 test_ratio=test_ratio, val_ratio=val_ratio,
                                 calibration_ratio=calibration_ratio, seed=seed, scale=scale,
                                 dim_to_reduce=dim_to_reduce, x_log_transform=x_log_transform, 
                                 x_scaler=scaler, y_log_transform=False, y_scaler=y_scaler, tail_test=test_size,
                                 pca = pca)
    data['cts'] = cts
    if scale:
        s_tr_x, s_tr_y = data['s_tr_x'], data['s_tr_y']

        def scale_x(x):
            return torch.Tensor(s_tr_x.transform(x))

        def scale_y(y):
            return torch.Tensor(s_tr_y.transform(y))

    else:
        def scale_x(x):
            return x

        def scale_y(y):
            return y

    data['scale_x'] = scale_x
    data['scale_y'] = scale_y
    return data
