import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    activation = 'selu'
    train = np.vstack([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_train.npy') for i in layers])
    pearson = np.abs(np.vstack([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]))
    spearman = np.abs(np.vstack([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers]))
    eig = np.vstack([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_eig.npy') for i in layers])

    # train_nol2 = [np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_train.npy') for i in layers]
    # pearson_nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_pearson.npy') for i in layers])
    # spearman_nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_spearman.npy') for i in layers])
    # eig_nol2 = [np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_eig.npy') for i in layers]
    activation = 'selu'
    train_vdp = [np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_train.npy') for i in layers]
    pearson_vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_pearson.npy') for i in layers], dtype=object)
    spearman_vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_spearman.npy') for i in layers])
    eig_vdp = [np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_eig.npy') for i in layers]

    # [sns.kdeplot(s, shade=False) for s in pearson]
    # plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    # plt.xlabel('Pearson Correlation')
    # plt.title('Deterministic Iris Pearson with random Inits')
    # plt.show()
    # plt.clf()
    # [sns.kdeplot(s, shade=False) for s in spearman]
    # plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    # plt.xlabel('Spearman Correlation')
    # plt.title('Deterministic Iris Spearman with random Inits')
    # plt.show()
    # plt.clf()
    # plt.scatter(train.reshape(-1), pearson.reshape(-1), label='Pearson')
    # plt.scatter(train.reshape(-1), spearman.reshape(-1), label='Spearman')
    # plt.legend()
    # plt.xlabel('Train Accuracy')
    # plt.ylabel('Correlation')
    # plt.title('Correlation as a function of train accuracy')
    # plt.show()
    # plt.clf()
    # plt.scatter(train[0], pearson[0], label='Pearson')
    # plt.scatter(train[0], spearman[0], label='Spearman')
    # plt.legend()
    # plt.xlabel('Train Accuracy')
    # plt.ylabel('Correlation')
    # plt.title('Correlation as a function of train accuracy')
    # plt.show()
    # plt.clf()

    plt.bar(np.arange(8)+1 - 0.2, np.mean(spearman, axis=1), 0.4, yerr=np.std(spearman, axis=1), label='Deterministic - SELU')
    mean = [np.mean(a[:, 0]) for a in spearman_vdp]
    std = [np.std(a[:, 0]) for a in spearman_vdp]
    # plt.bar(np.arange(8)+1 + 0.2, np.mean(spearman_vdp[:,:,0], axis=1), 0.4, yerr=np.std(spearman_vdp[:,:,0], axis=1), label='Stochastic - Tanh')
    plt.bar(np.arange(8)+1 + 0.2, mean, 0.4, yerr=std, label='Stochastic - SELU')
    plt.xticks(np.arange(8)+1, ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Number of Layers')
    plt.ylabel('Spearman Correlation')
    plt.title('Iris')
    plt.legend()
    plt.show()
    mean = [np.nanmean(a) for a in eig_vdp]
    std = [np.nanstd(a) for a in eig_vdp]
    plt.bar(np.arange(8)+1 - 0.2, np.nanmean(eig, axis=1), 0.4, yerr=np.nanstd(eig_vdp, axis=1), label='Deterministic - SELU')
    # plt.bar(np.arange(8)+1 + 0.2, np.nanmean(eig_vdp, axis=1), 0.4, yerr=np.nanstd(eig_vdp, axis=1), label='Stochastic - SELU')
    plt.bar(np.arange(8)+1 + 0.2, mean, 0.4, yerr=std, label='Stochastic - SELU')
    plt.xticks(np.arange(8)+1, ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Number of Layers')
    plt.ylabel('Top Eigenvalue of Hessian')
    plt.title('Iris')
    plt.legend()
    plt.show()


    # [sns.kdeplot(s, shade=False) for s in pearson_nol2]
    # plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    # plt.xlabel('Pearson Correlation')
    # plt.title('Deterministic Iris Pearson no weight decay with random Inits')
    # plt.show()
    # plt.clf()
    # [sns.kdeplot(s, shade=False) for s in spearman_nol2]
    # plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    # plt.xlabel('Spearman Correlation')
    # plt.title('Deterministic Iris Spearman no weight decay with random Inits')
    # plt.show()
    # plt.clf()
    # plt.scatter(train_nol2[0], pearson_nol2[0], label='Pearson')
    # plt.scatter(train_nol2[0], spearman_nol2[0], label='Spearman')
    # plt.legend()
    # plt.xlabel('Train Accuracy')
    # plt.ylabel('Correlation')
    # plt.title('Correlation as a function of train accuracy')
    # plt.show()
    # plt.clf()
    #
    # plt.bar(layers, [np.mean(s) for s in spearman_nol2], yerr=[np.std(s) for s in spearman_nol2])
    # plt.xlabel('Number of Layers')
    # plt.ylabel('Spearman Correlation')
    # plt.title('Deterministic Network with weight decay')
    # plt.show()

    # vdp_'+activation+' = np.load('legacy_dir/vdp_1l_'+activation+'_spearman.npy')
    # det_'+activation+' = np.load('figure1_l2_'+activation+'/det_1l_l2_approx_spearman.npy')
    # sns.kdeplot(np.abs(det_'+activation+'), shade=True)
    # sns.kdeplot(np.abs(vdp_'+activation+'[:, 0]), shade=True)
    # plt.legend(['Deterministic', 'Stochastic'])
    # plt.xlabel('Spearman Correlation')
    # plt.title('1-layer, '+activation+' Activations')
    # plt.show()
    # 
    # vdp_'+activation+' = np.load('legacy_dir/vdp_1l_'+activation+'_spearman.npy')
    # det_'+activation+' = np.load('figure1_l2_'+activation+'/det_1l_l2_spearman.npy')
    # sns.kdeplot(np.abs(det_'+activation+'), shade=True)
    # sns.kdeplot(np.abs(vdp_'+activation+'[:, 0]), shade=True)
    # plt.legend(['Deterministic', 'Stochastic'])
    # plt.xlabel('Spearman Correlation')
    # plt.title('1-layer, '+activation+' Activations')
    # plt.show()
    pass


if __name__ == '__main__':
    main()
