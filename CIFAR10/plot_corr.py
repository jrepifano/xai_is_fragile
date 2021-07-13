import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


est = np.load('est_loss_diffs_cifar.npy')
true = np.load('true_loss_diffs_cifar.npy')

pearson = [pearsonr(x, y)[0].item() for (x, y) in zip(est, true)]
spearman = [spearmanr(x, y)[0].item() for (x, y) in zip(est, true)]

sns.kdeplot(np.abs(pearson), label='Absolute Pearson')
sns.kdeplot(np.abs(spearman), label='Absolute Spearman')
plt.legend()
plt.xlabel('Correlation')
plt.title('Top 40 influential points')
plt.show()
print(np.max(np.abs(pearson)))
print(np.max(np.abs(spearman)))