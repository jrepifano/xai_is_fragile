import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


num_train = [5, 10, 20, 30, 40, 50]

est = np.load('lenet_results/est_loss_diffs_max.npy')
true = np.load('lenet_results/true_loss_diffs_max.npy')

est_swa = np.load('lenet_results/est_loss_diffs_max_swa.npy')
true_swa = np.load('lenet_results/true_loss_diffs_max_swa.npy')

df = pd.DataFrame(columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])
for i in num_train:
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est[:, :i], true[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['L2' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est_swa[:, :i], true_swa[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['L2 SWA' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)


df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'], errors='coerce')
sns.lineplot(x='Number of Training Examples', y='Spearman Correlation', data=df, hue='Type', ci=95)
plt.show()

plt.errorbar(np.arange(50), np.mean(true, axis=0), yerr=np.std(true, axis=0))
plt.show()
plt.errorbar(np.arange(50), np.mean(est, axis=0), yerr=np.std(est, axis=0))
plt.show()