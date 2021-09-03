import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


num_train = [5, 10, 20, 30, 40, 50]

est = np.load('results/lenet/est_loss_diffs_max.npy')
true = np.load('results/lenet/true_loss_diffs_max.npy')
plt.plot(est[0])
plt.show()
plt.plot(true[0])
plt.show()
est_swa = np.load('results/lenet/est_loss_diffs_med.npy')
true_swa = np.load('results/lenet/true_loss_diffs_med.npy')

est_vdp = np.load('results/lenet/est_loss_diffs_max_vdp.npy')
true_vdp = np.load('results/lenet/true_loss_diffs_max_vdp.npy')

est_swa_vdp = np.load('results/lenet/est_loss_diffs_med_vdp.npy')
true_swa_vdp = np.load('results/lenet/true_loss_diffs_med_vdp.npy')

df = pd.DataFrame(columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])
for i in num_train:
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est[:, :i], true[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['Det Max' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est_swa[:, :i], true_swa[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['Det Median' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)

    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est_vdp[:, :i], true_vdp[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['VDP Max' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est_swa_vdp[:, :i], true_swa_vdp[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['VDP Median' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)


df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'], errors='coerce')
sns.lineplot(x='Number of Training Examples', y='Spearman Correlation', data=df, hue='Type', ci=95)
plt.title('MNIST - LeNet')
plt.show()


est = np.load('results/vgg/est_loss_diffs_max_vgg.npy')
true = np.load('results/vgg/true_loss_diffs_max_vgg.npy')

est_swa = np.load('results/vgg/est_loss_diffs_med_vgg.npy')
true_swa = np.load('results/vgg/true_loss_diffs_med_vgg.npy')

df = pd.DataFrame(columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])
for i in num_train:
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est[:, :i], true[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['Det Max' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est_swa[:, :i], true_swa[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['Det Median' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'], errors='coerce')
sns.lineplot(x='Number of Training Examples', y='Spearman Correlation', data=df, hue='Type', ci=95)
plt.title('MNIST - VGG13')
plt.show()


# Last shot
num_train = [5, 10, 20, 30, 40]
est = np.load('legacy_results/est_loss_diffs_lastshot.npy')
true = np.load('legacy_results/true_loss_diffs_lastshot.npy')

df = pd.DataFrame(columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])
for i in num_train:
    spearman = [np.abs(spearmanr(a, b)[0]) for (a, b) in zip(est[:, :i], true[:, :i])]
    df = pd.concat((df, pd.DataFrame(np.vstack((spearman, i * np.ones_like(spearman),
                                                ['Det Max' for _ in range(len(spearman))])).T,
                                     columns=['Spearman Correlation', 'Number of Training Examples', 'Type'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'], errors='coerce')
sns.lineplot(x='Number of Training Examples', y='Spearman Correlation', data=df, hue='Type', ci=95)
plt.title('MNIST - Lenet "Last Shot" (more training I think)')
plt.show()
plt.plot(est[1])
plt.show()
plt.plot(true[1])
plt.show()
pass