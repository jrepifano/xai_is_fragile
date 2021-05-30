import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    train = np.vstack([np.load('det_' + str(i) + 'l_l2_approx_train.npy') for i in layers])
    test = np.vstack([np.load('det_' + str(i) + 'l_l2_approx_test.npy') for i in layers])
    pearson = np.abs(np.vstack([np.load('det_' + str(i) + 'l_l2_approx_pearson.npy') for i in layers]))
    spearman = np.abs(np.vstack([np.load('det_' + str(i) + 'l_l2_approx_spearman.npy') for i in layers]))
    eig = np.vstack([np.load('det_' + str(i) + 'l_l2_approx_eig.npy') for i in layers])

    [sns.kdeplot(s, shade=False) for s in pearson]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Pearson Correlation')
    plt.title('Iris Dataset Pearson Correlation with random Inits')
    plt.show()
    plt.clf()
    [sns.kdeplot(s, shade=False) for s in spearman]
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Spearman Correlation')
    plt.title('Iris Dataset Spearman Correlation with random Inits')
    plt.show()
    plt.clf()
    pass


if __name__ == '__main__':
    main()
