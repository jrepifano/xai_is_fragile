import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    layers = [1, 2, 3, 4, 5, 6, 7, 8]
    train = np.mean(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_train.npy') for i in layers]), axis=1)
    test = np.mean(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_test.npy') for i in layers]), axis=1)
    spearman = np.max(np.vstack([np.load('figure1/det_'+str(i)+'l_l2_spearman.npy') for i in layers]), axis=1)
    pearson = np.max(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]), axis=1)
    mean_spearman = np.mean(np.vstack([np.load('figure1/det_'+str(i)+'l_l2_spearman.npy') for i in layers]), axis=1)
    mean_pearson = np.mean(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]), axis=1)
    eig = np.mean(np.vstack([np.load('figure1/det_' + str(i) + 'l_l2_eig.npy') for i in layers]), axis=1)

    plt.plot(layers, train, label='Mean Train Acc')
    plt.plot(layers, test, label='Mean Test Acc')
    plt.legend()
    plt.xlabel('Number of Layers')
    plt.ylabel('Accuracy')
    plt.title('Iris Train and Test Accuracy')
    plt.show()
    plt.clf()
    plt.bar(layers, pearson)
    plt.xlabel('Number of Layers')
    plt.ylabel('Correlation')
    plt.title('Max Pearson Correlation')
    plt.show()
    plt.clf()
    plt.bar(layers, spearman)
    plt.xlabel('Number of Layers')
    plt.ylabel('Correlation')
    plt.title('Max Spearman Correlation')
    plt.show()
    plt.clf()
    plt.bar(layers, mean_pearson)
    plt.xlabel('Number of Layers')
    plt.ylabel('Correlation')
    plt.title('Mean Pearson Correlation')
    plt.show()
    plt.clf()
    plt.bar(layers, mean_spearman)
    plt.xlabel('Number of Layers')
    plt.ylabel('Correlation')
    plt.title('Mean Spearman Correlation')
    plt.show()
    plt.clf()
    plt.bar(layers, eig)
    plt.xlabel('Number of Layers')
    plt.title('Largest Hessian Eigenvalue')
    plt.show()
    plt.clf()



if __name__ == '__main__':
    main()
