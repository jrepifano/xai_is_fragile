import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    # train = np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_approx_train.npy') for i in layers])
    # test = np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_approx_test.npy') for i in layers])
    # pearson = np.abs(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_approx_pearson.npy') for i in layers]))
    # spearman = np.abs(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_approx_spearman.npy') for i in layers]))
    # eig = np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_approx_eig.npy') for i in layers])

    train = np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_train.npy') for i in layers])
    pearson = np.abs(np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]))
    spearman = np.abs(np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_spearman.npy') for i in layers]))
    eig = np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_eig.npy') for i in layers])

    layers = [1, 2, 3, 5, 6, 7, 8]
    train_nol2 = [np.load('figure1_nol2/det_' + str(i) + 'l_train.npy') for i in layers]
    test_nol2 = [np.load('figure1_nol2/det_' + str(i) + 'l_test.npy') for i in layers]
    pearson_nol2 = np.abs([np.load('figure1_nol2/det_' + str(i) + 'l_pearson.npy') for i in layers])
    spearman_nol2 = np.abs([np.load('figure1_nol2/det_' + str(i) + 'l_spearman.npy') for i in layers])
    eig_nol2 = [np.load('figure1_nol2/det_' + str(i) + 'l_eig.npy') for i in layers]

    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    vdp_train = [np.load('figure1_vdp/vdp_' + str(i) + 'l_train.npy') for i in layers]
    vdp_pearson = [np.load('figure1_vdp/vdp_' + str(i) + 'l_pearson.npy') for i in layers]
    vdp_spearman = [np.load('figure1_vdp/vdp_' + str(i) + 'l_spearman.npy') for i in layers]
    vdp_eig = [np.load('figure1_vdp/vdp_' + str(i) + 'l_eig.npy') for i in layers]


    [sns.kdeplot(s, shade=False) for s in pearson]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Pearson Correlation')
    plt.title('Deterministic Iris Pearson with random Inits')
    plt.show()
    plt.clf()
    [sns.kdeplot(s, shade=False) for s in spearman]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Spearman Correlation')
    plt.title('Deterministic Iris Spearman with random Inits')
    plt.show()
    plt.clf()
    plt.scatter(train.reshape(-1), pearson.reshape(-1), label='Pearson')
    plt.scatter(train.reshape(-1), spearman.reshape(-1), label='Spearman')
    plt.legend()
    plt.xlabel('Train Accuracy')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of train accuracy')
    plt.show()
    plt.clf()
    plt.scatter(train[0], pearson[0], label='Pearson')
    plt.scatter(train[0], spearman[0], label='Spearman')
    plt.legend()
    plt.xlabel('Train Accuracy')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of train accuracy')
    plt.show()
    plt.clf()

    [sns.kdeplot(s, shade=False) for s in pearson_nol2]
    plt.legend(['1', '2', '3', '5', '6', '7', '8'])
    plt.xlabel('Pearson Correlation')
    plt.title('Deterministic no decay Iris Pearson with random Inits')
    plt.show()
    plt.clf()
    [sns.kdeplot(s, shade=False) for s in spearman_nol2]
    plt.legend(['1', '2', '3', '5', '6', '7', '8'])
    plt.xlabel('Spearman Correlation')
    plt.title('Deterministic no decay Iris Spearman with random Inits')
    plt.show()
    plt.clf()
    # plt.scatter(train_nol2.reshape(-1), pearson_nol2.reshape(-1), label='Pearson')
    # plt.scatter(train_nol2.reshape(-1), spearman_nol2.reshape(-1), label='Spearman')
    # plt.legend()
    # plt.xlabel('Train Accuracy')
    # plt.ylabel('Correlation')
    # plt.title('Correlation no decay as a function of train accuracy')
    # plt.show()
    # plt.clf()
    plt.scatter(train_nol2[0], pearson_nol2[0], label='Pearson')
    plt.scatter(train_nol2[0], spearman_nol2[0], label='Spearman')
    plt.legend()
    plt.xlabel('Train Accuracy')
    plt.ylabel('Correlation')
    plt.title('Correlation no decay as a function of train accuracy')
    plt.show()
    plt.clf()


    sns.kdeplot([corr[0] for corr in vdp_pearson[0]])
    plt.legend(['1'])
    plt.xlabel('Pearson Correlation')
    plt.title('Stochastic Iris Pearson with random Inits')
    plt.show()
    plt.clf()
    sns.kdeplot([corr[0] for corr in vdp_spearman[0]])
    plt.legend(['1'])
    plt.xlabel('Pearson Correlation')
    plt.title('Stochastic Iris Spearman with random Inits')
    plt.show()
    plt.clf()
    plt.scatter(vdp_train[0], [corr[0] for corr in vdp_pearson[0]], label='Pearson')
    plt.scatter(vdp_train[0], [corr[0] for corr in vdp_spearman[0]], label='Spearman')
    plt.legend()
    plt.xlabel('Train Accuracy')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of train accuracy')
    plt.show()
    plt.clf()
    pass


if __name__ == '__main__':
    main()
