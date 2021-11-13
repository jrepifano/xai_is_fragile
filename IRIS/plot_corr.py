import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


layers = [1, 2, 3, 4, 5, 6, 7, 8]
widths = [8, 14, 20, 30, 40, 50]

activation = 'selu'
nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
nol2swa = np.abs([np.load('figure1_depth_nol2_'+activation+'_swa/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
l2 = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers]).T
l2swa = np.abs([np.load('figure1_depth_l2_'+activation+'_swa/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_spearman.npy') for i in layers])[:, :, 0].T

df = pd.DataFrame(columns=['Spearman Correlation', 'Type', 'Network Depth'])
for i in range(len(layers)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    # df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
    #                    np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
    #                    np.array([str(layers[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
    #                   columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'], errors='coerce')
sns.barplot(x='Network Depth', y='Spearman Correlation', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))
plt.ylim([0, 1.1])

plt.show()

activation = 'selu'
nol2 = np.abs([np.load('figure1_width_nol2_'+activation+'/det_' + str(i) + 'w_spearman.npy') for i in widths]).T
nol2swa = np.abs([np.load('figure1_width_nol2_'+activation+'_swa/det_' + str(i) + 'w_spearman.npy') for i in widths]).T
l2 = np.abs([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_spearman.npy') for i in widths]).T
l2swa = np.abs([np.load('figure1_width_l2_'+activation+'_swa/det_' + str(i) + 'w_spearman.npy') for i in widths]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_spearman.npy') for i in widths])[:, :, 0].T

df = pd.DataFrame(columns=['Spearman Correlation', 'Type', 'Layer Width'])
for i in range(len(widths)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    # df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
    #                    np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
    #                    np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
    #                   columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'])
sns.barplot(x='Layer Width', y='Spearman Correlation', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))
plt.ylim([0, 1.1])

plt.show()


activation = 'selu'
nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_pearson.npy') for i in layers]).T
nol2swa = np.abs([np.load('figure1_depth_nol2_'+activation+'_swa/det_' + str(i) + 'l_pearson.npy') for i in layers]).T
l2 = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_pearson.npy') for i in layers]).T
l2swa = np.abs([np.load('figure1_depth_l2_'+activation+'_swa/det_' + str(i) + 'l_pearson.npy') for i in layers]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_pearson.npy') for i in layers])[:, :, 0].T

df = pd.DataFrame(columns=['Pearson Correlation', 'Type', 'Network Depth'])
for i in range(len(layers)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Network Depth'])), ignore_index=True)
df['Pearson Correlation'] = pd.to_numeric(df['Pearson Correlation'], errors='coerce')
sns.barplot(x='Network Depth', y='Pearson Correlation', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))
plt.ylim([0, 1.1])

plt.show()

activation = 'selu'
nol2 = np.abs([np.load('figure1_width_nol2_'+activation+'/det_' + str(i) + 'w_pearson.npy') for i in widths]).T
nol2swa = np.abs([np.load('figure1_width_nol2_'+activation+'_swa/det_' + str(i) + 'w_pearson.npy') for i in widths]).T
l2 = np.abs([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_pearson.npy') for i in widths]).T
l2swa = np.abs([np.load('figure1_width_l2_'+activation+'_swa/det_' + str(i) + 'w_pearson.npy') for i in widths]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_pearson.npy') for i in widths])[:, :, 0].T

df = pd.DataFrame(columns=['Pearson Correlation', 'Type', 'Layer Width'])
for i in range(len(widths)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Pearson Correlation', 'Type', 'Layer Width'])), ignore_index=True)
df['Pearson Correlation'] = pd.to_numeric(df['Pearson Correlation'])
sns.barplot(x='Layer Width', y='Pearson Correlation', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))
plt.ylim([0, 1.1])

plt.show()

activation = 'selu'
nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_train.npy') for i in layers]).T
nol2swa = np.abs([np.load('figure1_depth_nol2_'+activation+'_swa/det_' + str(i) + 'l_train.npy') for i in layers]).T
l2 = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_train.npy') for i in layers]).T
l2swa = np.abs([np.load('figure1_depth_l2_'+activation+'_swa/det_' + str(i) + 'l_train.npy') for i in layers]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_train.npy') for i in layers]).T

df = pd.DataFrame(columns=['Train Accuracy', 'Type', 'Network Depth'])
for i in range(len(layers)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Network Depth'])), ignore_index=True)
df['Train Accuracy'] = pd.to_numeric(df['Train Accuracy'], errors='coerce')
sns.barplot(x='Network Depth', y='Train Accuracy', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))

plt.ylim([0.5, 1.5])
plt.show()

activation = 'selu'
nol2 = np.abs([np.load('figure1_width_nol2_'+activation+'/det_' + str(i) + 'w_train.npy') for i in widths]).T
nol2swa = np.abs([np.load('figure1_width_nol2_'+activation+'_swa/det_' + str(i) + 'w_train.npy') for i in widths]).T
l2 = np.abs([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_train.npy') for i in widths]).T
l2swa = np.abs([np.load('figure1_width_l2_'+activation+'_swa/det_' + str(i) + 'w_train.npy') for i in widths]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_train.npy') for i in widths]).T

df = pd.DataFrame(columns=['Train Accuracy', 'Type', 'Layer Width'])
for i in range(len(widths)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Train Accuracy', 'Type', 'Layer Width'])), ignore_index=True)
df['Train Accuracy'] = pd.to_numeric(df['Train Accuracy'], errors='coerce')
sns.barplot(x='Layer Width', y='Train Accuracy', hue='Type', data=df, ci=95)
plt.legend(loc="upper left", ncol=len(df.columns))

plt.ylim([0.5, 1.5])
plt.show()

activation = 'selu'
nol2 = np.array([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_eig.npy') for i in layers]).T
nol2swa = np.array([np.load('figure1_depth_nol2_'+activation+'_swa/det_' + str(i) + 'l_eig.npy') for i in layers]).T
l2 = np.array([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_eig.npy') for i in layers]).T
l2swa = np.array([np.load('figure1_depth_l2_'+activation+'_swa/det_' + str(i) + 'l_eig.npy') for i in layers]).T
activation = 'tanh'
vdp = np.array([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_eig.npy') for i in layers]).T

df = pd.DataFrame(columns=['Top Eigenvalue', 'Type', 'Network Depth'])
for i in range(len(layers)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Network Depth'])), ignore_index=True)
    # df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
    #                    np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
    #                    np.array([str(layers[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
    #                   columns=['Top Eigenvalue', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(layers[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Network Depth'])), ignore_index=True)
df['Top Eigenvalue'] = pd.to_numeric(df['Top Eigenvalue'], errors='coerce')
g = sns.barplot(x='Network Depth', y='Top Eigenvalue', hue='Type', data=df, ci=95)
g.set_yscale("log")
plt.legend(loc="upper left", ncol=len(df.columns))

plt.show()

activation = 'selu'
nol2 = np.array([np.load('figure1_width_nol2_'+activation+'/det_' + str(i) + 'w_eig.npy') for i in widths]).T
nol2swa = np.array([np.load('figure1_width_nol2_'+activation+'_swa/det_' + str(i) + 'w_eig.npy') for i in widths]).T
l2 = np.array([np.load('figure1_width_l2_'+activation+'/det_' + str(i) + 'w_l2_eig.npy') for i in widths]).T
l2swa = np.array([np.load('figure1_width_l2_'+activation+'_swa/det_' + str(i) + 'w_eig.npy') for i in widths]).T
activation = 'tanh'
vdp = np.array([np.load('figure1_width_vdp_'+activation+'/vdp_' + str(i) + 'w_eig.npy') for i in widths]).T

df = pd.DataFrame(columns=['Top Eigenvalue', 'Type', 'Layer Width'])
for i in range(len(widths)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No L2' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No L2 SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['L2' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Layer Width'])), ignore_index=True)
    # df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
    #                    np.array(['L2 SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
    #                    np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
    #                   columns=['Top Eigenvalue', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['BNN' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Top Eigenvalue', 'Type', 'Layer Width'])), ignore_index=True)
df['Top Eigenvalue'] = pd.to_numeric(df['Top Eigenvalue'], errors='coerce')
g = sns.barplot(x='Layer Width', y='Top Eigenvalue', hue='Type', data=df, ci=95)
g.set_yscale("log")
plt.legend(loc="upper left", ncol=len(df.columns))

plt.show()