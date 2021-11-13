import os
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    activation = 'selu'
    train = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_train.npy') for i in layers]
    pearson = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_pearson.npy') for i in layers])
    spearman = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers])
    eig = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_eig.npy') for i in layers]

    activation = 'tanh'
    train_vdp = [np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_train.npy') for i in layers]
    pearson_vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_pearson.npy') for i in layers], dtype=object)
    spearman_vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_spearman.npy') for i in layers])
    eig_vdp = [np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_eig.npy') for i in layers]

    mean = [np.mean(a) for a in spearman]
    std = [np.std(a) for a in spearman]
    plt.bar(np.arange(8)+1 - 0.2, mean, 0.4, yerr=std, label='L2')
    mean = [np.mean(a[:, 0]) for a in spearman_vdp]
    std = [np.std(a[:, 0]) for a in spearman_vdp]
    plt.bar(np.arange(8)+1 + 0.2, mean, 0.4, yerr=std, label='BNN')
    plt.xticks(np.arange(8)+1, ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Network Depth')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.show()
    mean = [np.nanmean(np.abs(a)) for a in eig]
    std = [np.nanstd(np.abs(a)) for a in eig]
    plt.bar(np.arange(8)+1 - 0.2, mean, 0.4, yerr=std, label='L2')
    mean = [np.nanmean(np.abs(a)) for a in eig_vdp]
    std = [np.nanstd(np.abs(a)) for a in eig_vdp]
    plt.bar(np.arange(8)+1 + 0.2, mean, 0.4, yerr=std, label='BNN')
    plt.xticks(np.arange(8)+1, ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Network Depth')
    plt.ylabel('Top Eigenvalue of Hessian')
    plt.legend()
    plt.show()

    widths = [8, 14, 20, 30, 40, 50]
    activation = 'selu'
    train = np.vstack([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_train.npy') for i in widths])
    pearson = np.abs(np.vstack([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_pearson.npy') for i in widths]))
    spearman = np.abs(np.vstack([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_spearman.npy') for i in widths]))
    eig = np.vstack([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_eig.npy') for i in widths])

    activation = 'tanh'
    train_vdp = [np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_train.npy') for i in widths]
    pearson_vdp = np.abs([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_pearson.npy') for i in widths], dtype=object)
    spearman_vdp = np.abs([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_spearman.npy') for i in widths])
    eig_vdp = [np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_eig.npy') for i in widths]

    plt.bar(np.arange(6)+1 - 0.2, np.mean(spearman, axis=1), 0.4, yerr=np.std(spearman, axis=1), label='L2')
    mean = [np.mean(a[:, 0]) for a in spearman_vdp]
    std = [np.std(a[:, 0]) for a in spearman_vdp]
    plt.bar(np.arange(6)+1 + 0.2, mean, 0.4, yerr=std, label='BNN')
    plt.xticks(np.arange(6)+1, ['8', '14', '20', '30', '40', '50'])
    plt.xlabel('Layer Width')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.show()
    mean = [np.nanmean(np.abs(a)) for a in eig_vdp]
    std = [np.nanstd(np.abs(a)) for a in eig_vdp]
    plt.bar(np.arange(6)+1 - 0.2, np.nanmean(eig, axis=1), 0.4, yerr=np.nanstd(eig, axis=1), label='L2')
    plt.bar(np.arange(6)+1 + 0.2, mean, 0.4, yerr=std, label='BNN')
    plt.xticks(np.arange(6)+1, ['8', '14', '20', '30', '40', '50'])
    plt.xlabel('Layer Width')
    plt.ylabel('Top Eigenvalue of Hessian')
    plt.legend()
    plt.show()

    activation = 'selu'
    train = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_train.npy') for i in layers]
    pearson = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_pearson.npy') for i in layers])
    spearman = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers])
    eig = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_eig.npy') for i in layers]

    mean = [np.mean(a) for a in spearman]
    std = [np.std(a) for a in spearman]
    plt.bar(np.arange(8)+1 - 0.2, mean, 0.4, yerr=std, label='Retrain all layers')

    activation = 'selu_frozen'
    train = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_train.npy') for i in layers]
    pearson = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_pearson.npy') for i in layers])
    spearman = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers])
    eig = [np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_eig.npy') for i in layers]

    mean = [np.mean(a) for a in spearman]
    std = [np.std(a) for a in spearman]
    plt.bar(np.arange(8)+1 + 0.2, mean, 0.4, yerr=std, label='Retrain Top Layer')
    plt.xticks(np.arange(8)+1, ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Network Depth')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
