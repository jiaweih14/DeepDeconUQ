"""
Part of the code is taken from https://github.com/Shai128/mqr
"""

import matplotlib
import pandas as pd
import numpy as np
import torch as torch
from matplotlib import patches, pyplot as plt
import logging
import helper
import os
import random
from sklearn.manifold import TSNE
logger = logging.getLogger(__name__)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate_performance(model, dataset_name, x_test, y_te, args):
    alpha = args.tau
    quantiles = torch.Tensor([alpha / 2, 1 - alpha / 2])

    with torch.no_grad():
        model_pred, orig = model.predict_q(
            x_test, quantiles, ens_pred_type='conf',
            recal_model=None, recal_type=None, return_orig=True
        )
    np.save(os.path.join(args.out_dir, 'orig_pred.npy'), orig)
    # Save the parameters and coverages for later use
    is_1d = '1d_' in args.dataset_name
    dataset_name = args.dataset_name.replace("1d_", "")
    if dataset_name.startswith('aml') or dataset_name.startswith('pbmc'):
        subject = dataset_name[4:]
        if not subject:
            subject = dataset_name + '_whole'
        outdir = args.out_dir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        if is_1d:
            preds = model_pred.squeeze().cpu().numpy()
            df = pd.DataFrame(np.concatenate([y_te.cpu().numpy(), preds], axis=1), columns=['truth', 'lower', 'upper'])
            df.to_csv(outdir+'NVQR_preds.csv')
        else:
            y_upper = model_pred[:, 1].cpu().numpy()
            y_lower = model_pred[:, 0].cpu().numpy()
            truth = y_te.cpu().numpy()
            # print(truth.shape, y_upper.shape, y_lower.shape)
            # print(np.concatenate([truth, y_lower, y_upper], axis=1).shape)
            cts = list(args.cts)
            columns = cts + [ct+'_lower' for ct in cts] + [ct+'_upper' for ct in cts]
            # print(f'length of columns: {len(columns)}, {columns}')
            df = pd.DataFrame(np.concatenate([truth, y_lower, y_upper], axis=1), columns=columns)
            df.to_csv(outdir+'NVQR_preds.csv')




n_clusters_plotted = 0
def plot_quantile_region(Y, y_lower, y_upper, is_real, is_marginal, full_y_grid, args, title_begin='',
                         save_fig_path=None):
    if args.suppress_plots or Y.shape[1] != 2:
        return
    global n_clusters_plotted

    if is_real or is_marginal:
        title = None  # title_begin + 'Quantile region'
        y_grid_repeated = full_y_grid.unsqueeze(1).repeat(1, len(y_upper), 1)
        in_region_idx = ((y_grid_repeated <= y_upper) & (y_grid_repeated >= y_lower)).float().prod(dim=-1).bool()
        in_region_points = y_grid_repeated[in_region_idx]
        idx = np.random.permutation(len(in_region_points))[:50000]
        in_region_points = in_region_points[idx]
        in_region_points = helper.filter_outlier_points(in_region_points)
        if n_clusters_plotted > 0:
            a_label = None
            b_label = None
        else:
            a_label = 'samples'
            b_label = 'quantile region'
        plot_samples(Y.cpu(), a_label, args, b=in_region_points.cpu(), b_label=b_label, a_color='b', axis_name='Y',
                     b_color='r', title=title, save_fig_path=save_fig_path, legend_place='lower left', b_alpha=0.015)
        if not is_marginal:
            n_clusters_plotted += 1
    else:
        title = None  # title_begin + 'Quantile contour'
        y_upper = y_upper.flatten().cpu()
        y_lower = y_lower.flatten().cpu()
        Y = Y.cpu()

        rect = patches.Rectangle((y_lower[0], y_lower[1]), (y_upper - y_lower)[0], (y_upper - y_lower)[1],
                                 linewidth=1, edgecolor='r', facecolor='r', label='quantile region', alpha=0.6)
        # Add the patch to the Axes
        figure = plt.figure()
        ax = figure.add_subplot()
        ax.add_patch(rect)
        ax.scatter(Y[:, 0], Y[:, 1], c='b', label='samples')
        plt.xlim([-2.5, 2.5])
        plt.ylim([0.25, 2.2])

        if n_clusters_plotted == 0:
            plt.legend('lower left')
            lines1 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='quantile region',
                                             markerfacecolor='red',
                                             markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            lines2 = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='samples',
                                             markerfacecolor='blue',
                                             markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            legend = plt.legend(handles=[lines1, lines2], loc='lower left')

            for lh in legend.legendHandles:
                lh.set_alpha(1)

        plt.xlabel("Y0")
        plt.ylabel("Y1")
        if title is not None:
            plt.title(title)
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        n_clusters_plotted += 1


def plot_samples(a, a_label=None, args=None, b=None, b_label=None, title=None, axis_name='X', a_color='b', b_color='g',
                 save_fig_path=None, a_alpha=1., b_alpha=1.,
                 a_radius=None, b_radius=None, legend_place='best', x_lim=None, y_lim=None, z_lim=None, legend_color='w', font_size=17):
    if args is not None and args.suppress_plots:
        return
    figure = plt.figure()
    matplotlib.rc('font', **{'size': font_size})
    if b is not None and len(b) == 0:
        b = None
    if b is not None:
        assert a.shape[1] == b.shape[1]
    if a.shape[1] > 5:
        return
    if a.shape[1] > 3:
        idx1 = np.random.permutation(len(a))[:1000]
        a = TSNE(n_components=3).fit_transform(a[idx1])
        if b is not None:
            idx2 = np.random.permutation(len(b))[:1000]
            b = TSNE(n_components=3).fit_transform(b[idx2])

    plot_params = {}
    a_plot_params = {
        'color': a_color,
        'alpha': a_alpha,
        **plot_params
    }
    if a_label is not None:
        a_plot_params['label'] = a_label

    if b is not None:
        b_plot_params = {
            'color': b_color,
            'alpha': b_alpha,
            **plot_params
        }
        if b_label is not None:
            b_plot_params['label'] = b_label

    if a.shape[1] == 3:
        ax = figure.add_subplot(projection='3d')

        if a_radius is not None:
            plt_sphere(figure, ax, a.numpy(), a_radius, a_color, a_alpha)

        else:
            ax.scatter(a[:, 0], a[:, 1], zs=a[:, 2], **a_plot_params)
        a_legend = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=a_label,
                                           markerfacecolor=a_color,
                                           markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
        if b is not None:

            if b_radius is not None:
                plt_sphere(figure, ax, b.numpy(), b_radius, b_color, b_alpha)

            else:
                ax.scatter(b[:, 0], b[:, 1], zs=b[:, 2], **b_plot_params)
            b_legend = matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=b_label,
                                               markerfacecolor=b_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
        handles = []
        if a_label is not None:
            handles += [a_legend]
        if b_label is not None:
            handles += [b_legend]
        if len(handles) > 0:
            plt.legend(handles=handles, loc=legend_place)

        ax.set_xlabel(axis_name + '0')
        ax.set_ylabel(axis_name + '1')
        ax.set_zlabel(axis_name + '2')


    elif a.shape[1] == 2:
        ax = figure.add_subplot()

        if a_radius is not None:
            circles = [plt.Circle(point, radius=a_radius, linewidth=0) for point in zip(*a.split(1, dim=1))]
            c = matplotlib.collections.PatchCollection(circles, **a_plot_params)

            a_legend = matplotlib.lines.Line2D([0], [0], marker='o', color=legend_color, label=a_label,
                                               markerfacecolor=a_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)
            ax.add_collection(c)
        else:
            a_legend = ax.scatter(a[:, 0], a[:, 1], **a_plot_params)

        if a_label is not None:
            plt.legend(handles=[a_legend], loc=legend_place)

        if b is not None:

            if b_radius is not None:
                circles = [plt.Circle(point, radius=b_radius, linewidth=0) for point in zip(*b.split(1, dim=1))]
                c = matplotlib.collections.PatchCollection(circles, **b_plot_params)
                ax.add_collection(c)
            else:
                ax.scatter(b[:, 0], b[:, 1], **b_plot_params)
            b_legend = matplotlib.lines.Line2D([0], [0], marker='o', color=legend_color, label=b_label,
                                               markerfacecolor=b_color,
                                               markersize=matplotlib.rcParams['lines.markersize'] * 1.3)

            handles = []
            if a_label is not None:
                handles += [a_legend]
            if b_label is not None:
                handles += [b_legend]
            if len(handles) > 0:
                plt.legend(handles=handles, loc=legend_place)

        ax.set_xlabel(axis_name + '0')
        ax.set_ylabel(axis_name + '1')

    else:
        ax = figure.add_subplot()
        ax.scatter(a[:, 0], np.zeros(len(a)), **a_plot_params)
        if b is not None:
            ax.scatter(b[:, 0], np.zeros(len(b)), **b_plot_params)
        plt.legend(loc=legend_place)
        ax.set_xlabel(axis_name)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)

    if title is not None:
        ax.set_title(title)
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')

    plt.show()

def plt_sphere(fig, ax, list_center, radius, color, alpha):
    ax.scatter(list_center[:, 0], list_center[:, 1],list_center[:, 2], color=color, alpha=alpha)