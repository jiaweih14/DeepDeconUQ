"""
Part of the code is taken from https://github.com/Shai128/mqr
"""

import sys, math
from argparse import Namespace
from copy import deepcopy
import numpy as np
from scipy.stats import norm as norm_distr
from scipy.stats import t as t_distr
from scipy.interpolate import interp1d
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import logging

logger = logging.getLogger(__name__)

def multivariate_qr_loss(model, y, x, q_list, device, args):
    num_pts = y.size(0)

    with torch.no_grad():
        l_list = torch.min(torch.stack([q_list, 1 - q_list], dim=1), dim=1)[0].to(device)
        u_list = 1.0 - l_list

    q_list = torch.cat([l_list, u_list], dim=0)
    num_q = q_list.shape[0]

    q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1).to(device)
    y_stacked = y.repeat(num_q, 1)

    if x is None:
        model_in = torch.cat([l_list, u_list], dim=0)
    else:
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)

    pred_y = model(model_in)

    diff = pred_y - y_stacked
    mask = (diff.ge(0).float() - q_rep).detach()  # / q_rep

    pinball_loss = (mask * diff).mean(dim=0).sum()

    return pinball_loss


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, use_bn=True,
                 actv_type='relu', dropout=0):
        super(LinearLayer, self).__init__()

        """ linear layer """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif actv_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError


        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)


    def reset_parameters(self):
        # # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        # if (reset_indv_bias is None) or (reset_indv_bias is False):
        #     init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        # if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
        #     init.constant_(self.bias, 0)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # concat channels and length of signal if input is from conv layer
        if len(input.shape) > 2:
            batch_size = input.shape[0]
            input = input.view(batch_size, -1)

        out = F.linear(input, self.weight, self.bias)
        #print('after matmul\n', out)

        if self.bn:
            out = self.bn(out)
        #print('after bn\n', out)
        if self.activation is not None:
            out = self.activation(out)

        if self.dropout > 0:
            out = self.dropout_layer(out)


        #print('after linear layer\n', out)

        return out


class vanilla_nn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_dimensions=[64, 64, 64], 
                 use_bn=False, actv_type='relu',
                 softmax=False, dropout=0, sigmoid=False, loss='mse'):

        super(vanilla_nn, self).__init__()
        self.softmax = softmax
        self.sigmoid = sigmoid
        # TODO: make loss an option
        if loss == 'mse':
            self.loss = nn.MSELoss()
        # elif loss == 'coverage':
        #     self.loss = CoverageLoss()
        # elif loss == 'quantile':
        #     self.loss = AllQuantileLoss()

        self.fcs = nn.ModuleList()
        """ input layer """
        """ input layer """
        self.fcs.append(LinearLayer(input_size, hidden_dimensions[0], bias,
                                    use_bn=use_bn, actv_type=actv_type, dropout=dropout))

        for i in range(0, len(hidden_dimensions)-1):
            self.fcs.append(LinearLayer(hidden_dimensions[i], hidden_dimensions[i+1], bias,
                                        use_bn=use_bn, actv_type=actv_type, dropout=dropout))

        self.fcs.append(LinearLayer(hidden_dimensions[-1], output_size, bias,
                                    use_bn=False, actv_type=None, dropout=0))

    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        elif self.sigmoid:
            out = F.sigmoid(X)
        else:
            out = X

        return out


class uq_model(object):

    def predict(self):
        raise NotImplementedError('Abstract Method')


""" QModelEns Utils """


def gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def get_ens_pred_interp(unc_preds, taus, fidelity=10000):
    """
    unc_preds 3D ndarray (ens_size, 99, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    # taus = np.arange(0.01, 1, 0.01)
    y_min, y_max = np.min(unc_preds), np.max(unc_preds)
    y_grid = np.linspace(y_min, y_max, fidelity)
    new_quants = []
    avg_cdfs = []
    for x_idx in tqdm.tqdm(range(unc_preds.shape[-1])):
        x_cdf = []
        for ens_idx in range(unc_preds.shape[0]):
            xs, ys = [], []
            targets = unc_preds[ens_idx, :, x_idx]
            for idx in np.argsort(targets):
                if len(xs) != 0 and targets[idx] <= xs[-1]:
                    continue
                xs.append(targets[idx])
                ys.append(taus[idx])
            intr = interp1d(xs, ys,
                            kind='linear',
                            fill_value=([0], [1]),
                            bounds_error=False)
            x_cdf.append(intr(y_grid))
        x_cdf = np.asarray(x_cdf)
        avg_cdf = np.mean(x_cdf, axis=0)
        avg_cdfs.append(avg_cdf)
        t_idx = 0
        x_quants = []
        for idx in range(len(avg_cdf)):
            if t_idx >= len(taus):
                break
            if taus[t_idx] <= avg_cdf[idx]:
                x_quants.append(y_grid[idx])
                t_idx += 1
        while t_idx < len(taus):
            x_quants.append(y_grid[-1])
            t_idx += 1
        new_quants.append(x_quants)
    return np.asarray(new_quants).T


def get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95, score_distr='z'):
    """
    unc_preds 3D ndarray (ens_size, num_tau, num_x)
    where for each ens_member, each row corresonds to tau 0.01, 0.02...
    and the columns are for the set of x being predicted over.
    """
    num_ens, num_tau, num_x = unc_preds.shape
    len_tau = taus.size

    mean_pred = np.mean(unc_preds, axis=0)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred / np.sqrt(num_ens)
    alpha = (1 - conf_level)  # is (1-C)

    # determine coefficient
    if score_distr == 'z':
        crit_value = norm_distr.ppf(1 - (0.5 * alpha))
    elif score_distr == 't':
        crit_value = t_distr.ppf(q=1 - (0.5 * alpha), df=(num_ens - 1))
    else:
        raise ValueError('score_distr must be one of z or t')

    gt_med = (taus > 0.5).reshape(-1, num_x)
    lt_med = ~gt_med
    assert gt_med.shape == mean_pred.shape == stderr_pred.shape
    out = (lt_med * (mean_pred - (float(crit_value) * stderr_pred)) +
           gt_med * (mean_pred + (float(crit_value) * stderr_pred))).T
    out = torch.from_numpy(out)
    return out


class QModelEns(uq_model):

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None, softmax_output=False):

        self.num_ens = num_ens
        self.device = device
        self.dropout = dropout
        # output_size  = 1
        if nn_input_size is None:
            nn_input_size = input_size + y_size

        self.model = [vanilla_nn(input_size=nn_input_size, output_size=output_size,
                                 hidden_dimensions=hidden_dimensions,
                                 dropout=dropout, softmax=softmax_output).to(device)
                      for _ in range(num_ens)]
        self.optimizers = [torch.optim.Adam(x.parameters(),
                                            lr=lr, weight_decay=wd)
                           for x in self.model]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [None for _ in range(num_ens)]
        self.best_va_ep = [0 for _ in range(num_ens)]
        self.done_training = False
        self.is_conformalized = False

    def use_device(self, device):
        self.device = device
        for idx in range(len(self.best_va_model)):
            self.best_va_model[idx] = self.best_va_model[idx].to(device)

        if device.type == 'cuda':
            for idx in range(len(self.best_va_model)):
                assert next(self.best_va_model[idx].parameters()).is_cuda

    def print_device(self):
        device_list = []
        for idx in range(len(self.best_va_model)):
            if next(self.best_va_model[idx].parameters()).is_cuda:
                device_list.append('cuda')
            else:
                device_list.append('cpu')
        print(device_list)

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y, x, q_list, self.device, args)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx], y, x,
                                             q_list, self.device, args)
                ens_loss.append(loss.cpu().item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def loss_boot(self, loss_fn, x_list, y_list, q_list, batch_q, take_step, args):
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y_list[idx], x_list[idx],
                                   q_list, self.device, args)
                else:
                    loss = gather_loss_per_q(loss_fn, self.model[idx],
                                             y_list[idx], x_list[idx], q_list,
                                             self.device, args)
                ens_loss.append(loss.detach().cpu().item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)

    def update_va_loss(self, loss_fn, x, y, q_list, batch_q, curr_ep, num_wait, args):
        with torch.no_grad():
            va_loss = self.loss(loss_fn, x, y, q_list, batch_q, take_step=False, args=args)

        # if torch.isnan(va_loss):
        #     print("va loss is nan!")

        for idx in range(self.num_ens):
            if self.keep_training[idx]:
                if va_loss[idx] < self.best_va_loss[idx]:
                    self.best_va_loss[idx] = va_loss[idx]
                    self.best_va_ep[idx] = curr_ep
                    self.best_va_model[idx] = deepcopy(self.model[idx])
                else:
                    if curr_ep - self.best_va_ep[idx] > num_wait:
                        self.keep_training[idx] = False

        if not any(self.keep_training):
            self.done_training = True

        return va_loss

    #####
    def predict(self, cdf_in, conf_level=0.95, score_distr='z',
                recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Only pass in cdf_in into model and return output
        If self is an ensemble, return a conservative output based on conf_bound
        specified by conf_level

        :param cdf_in: tensor [x, p], of size (num_x, dim_x + 1)
        :param conf_level: confidence level for ensemble prediction
        :param score_distr: 'z' or 't' for confidence bound coefficient
        :param recal_model:
        :param recal_type:
        :return:
        """

        if self.num_ens == 1:
            with torch.no_grad():
                if use_best_va_model:
                    pred = self.best_va_model[0](cdf_in)
                else:
                    pred = self.model[0](cdf_in)
        if self.num_ens > 1:
            pred_list = []
            if use_best_va_model:
                models = self.best_va_model
            else:
                models = self.model

            for m in models:
                with torch.no_grad():
                    pred_list.append(m(cdf_in).T.unsqueeze(0))

            unc_preds = torch.cat(pred_list, dim=0).detach().cpu().numpy()  # shape (num_ens, num_x, 1)
            taus = cdf_in[:, -1].flatten().cpu().numpy()
            pred = get_ens_pred_conf_bound(unc_preds, taus, conf_level=0.95,
                                           score_distr='z')
            pred = pred.to(cdf_in.device)

        return pred

    #####

    def predict_q(self, x, q_list=None, ens_pred_type='conf',
                  recal_model=None, recal_type=None, use_best_va_model=True):
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param ens_pred_type:
        :param recal_model:
        :param recal_type:
        :return:
        """

        if q_list is None:
            q_list = torch.arange(0.01, 0.99, 0.01)
        else:
            q_list = q_list.flatten()

        if self.num_ens > 1:
            # choose function to make ens predictions
            if ens_pred_type == 'conf':
                ens_pred_fn = get_ens_pred_conf_bound
            elif ens_pred_type == 'interp':
                ens_pred_fn = get_ens_pred_interp
            else:
                raise ValueError('ens_pred_type must be one of conf or interp')

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []
        for p in q_list:
            if recal_model is not None:
                if recal_type == 'torch':
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == 'sklearn':
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError('recal_type incorrect')
            else:
                in_p = float(p)
            p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1).to(x.device)

            cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            cdf_pred = self.predict(cdf_in, use_best_va_model=use_best_va_model)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred.unsqueeze(1))

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q, y_shape[1])
        # assert pred_mat.shape == (num_x, num_q)
        return pred_mat

class MultivariateQuantileModel(QModelEns):

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None, y_grid_size=3e3, z_grid_size=4e4, softmax_output=False, args = None):
        QModelEns.__init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                           num_ens, device, output_size, nn_input_size, softmax_output)
        self.y_grid_size = y_grid_size
        self.z_grid_size = z_grid_size
        self.is_conformalized = False
        self.args = args

    def get_coverage_identifiers(self, Y, y_lower, y_upper):
        return ((Y <= y_upper) & (Y >= y_lower)).float().prod(dim=1).bool()

    def conformalize(self, x_cal, y_cal, conformalization_tau, tau):

        quantiles = torch.Tensor([tau / 2, 1 - tau / 2])
        model_pred = self.predict_q(
            x_cal, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None
        )
        y_upper = model_pred[:, 1]
        y_lower = model_pred[:, 0]

        distance_from_boundaries = torch.zeros(len(y_cal), 2 * y_cal.shape[1]).to(y_cal.device)
        for dim in range(y_cal.shape[1]):
            distance_from_boundaries[:, dim * 2] = y_lower[:, dim] - y_cal[:, dim]
            distance_from_boundaries[:, dim * 2 + 1] = y_cal[:, dim] - y_upper[:, dim]
        scores = distance_from_boundaries.max(dim=1)[0]
        n = len(scores)
        q = min(1.0, np.ceil((n + 1) * (1 - conformalization_tau)) / n)
        Q = torch.quantile(scores, q=q)
        logger.info(f"q: {q}, Q: {Q}, tail score: {sorted(scores.tolist())[-10:]}")
        
        self.correction = torch.ones(y_cal.shape[1]).to(y_cal.device) * Q
        self.radius = Q
        self.is_conformalized = True
        return scores

    def predict_q(self, x, q_list=None, ens_pred_type='conf',
                  recal_model=None, recal_type=None, use_best_va_model=True, return_orig = False):

        pred = super().predict_q(x, q_list, ens_pred_type, recal_model, recal_type, use_best_va_model)
        orig = pred.detach().clone().cpu().numpy()
        if self.is_conformalized:
            pred[:, 1] = torch.clamp(pred[:, 1] + self.correction, 0.0, 1.0)  # upper quantile
            pred[:, 0] = torch.clamp(pred[:, 0] - self.correction, 0.0, 1.0)  # lower quantile
        if return_orig:
            return pred, orig
        return pred

