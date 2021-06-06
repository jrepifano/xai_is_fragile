import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]

    train = np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_train.npy') for i in layers])
    pearson = np.abs(np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]))
    spearman = np.abs(np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_spearman.npy') for i in layers]))
    eig = np.vstack([np.load('figure1_l2_selu/det_' + str(i) + 'l_l2_eig.npy') for i in layers])

    train_nol2 = [np.load('figure1_nol2_selu/det_' + str(i) + 'l_train.npy') for i in layers]
    pearson_nol2 = np.abs([np.load('figure1_nol2_selu/det_' + str(i) + 'l_pearson.npy') for i in layers])
    spearman_nol2 = np.abs([np.load('figure1_nol2_selu/det_' + str(i) + 'l_spearman.npy') for i in layers])
    eig_nol2 = [np.load('figure1_nol2_selu/det_' + str(i) + 'l_eig.npy') for i in layers]

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

    plt.bar(layers, np.mean(spearman, axis=1), yerr=np.std(spearman, axis=1))
    plt.xlabel('Number of Layers')
    plt.ylabel('Spearman Correlation')
    plt.title('Deterministic Network with weight decay')
    plt.show()


    [sns.kdeplot(s, shade=False) for s in pearson_nol2]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Pearson Correlation')
    plt.title('Deterministic Iris Pearson no weight decay with random Inits')
    plt.show()
    plt.clf()
    [sns.kdeplot(s, shade=False) for s in spearman_nol2]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Spearman Correlation')
    plt.title('Deterministic Iris Spearman no weight decay with random Inits')
    plt.show()
    plt.clf()
    plt.scatter(train_nol2[0], pearson_nol2[0], label='Pearson')
    plt.scatter(train_nol2[0], spearman_nol2[0], label='Spearman')
    plt.legend()
    plt.xlabel('Train Accuracy')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of train accuracy')
    plt.show()
    plt.clf()

    plt.bar(layers, [np.mean(s) for s in spearman_nol2], yerr=[np.std(s) for s in spearman_nol2])
    plt.xlabel('Number of Layers')
    plt.ylabel('Spearman Correlation')
    plt.title('Deterministic Network with weight decay')
    plt.show()

    pass


if __name__ == '__main__':
    main()
