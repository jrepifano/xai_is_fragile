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


layers = [1, 2, 3, 4, 5, 6, 8]
widths = [8, 20, 14, 30, 40, 50]

activation = 'selu'
nol2 = np.abs([np.load('figure1_nol2_'+activation+'/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
nol2swa = np.abs([np.load('figure1_depth_nol2_'+activation+'_swa/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
l2 = np.abs([np.load('figure1_l2_'+activation+'/det_' + str(i) + 'l_l2_spearman.npy') for i in layers]).T
l2swa = np.abs([np.load('figure1_depth_l2_'+activation+'_swa/det_' + str(i) + 'l_spearman.npy') for i in layers]).T
activation = 'tanh'
vdp = np.abs([np.load('figure1_vdp_'+activation+'/vdp_' + str(i) + 'l_spearman.npy') for i in layers])[:, :, 0].T

df = pd.DataFrame(columns=['Spearman Correlation', 'Type', 'Network Depth'])
for i in range(len(widths)):
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2[:, i].reshape(-1, 1),
                       np.array(['No Weight Decay' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No Weight Decay w/SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['Weight Decay' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['Weight Decay w/SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['VDP' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Network Depth'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'])
sns.barplot(x='Network Depth', y='Spearman Correlation', hue='Type', data=df, ci=95)
plt.title('Iris Dataset')
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
                       np.array(['No Weight Decay' for _ in range(len(nol2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((nol2swa[:, i].reshape(-1, 1),
                       np.array(['No Weight Decay w/SWA' for _ in range(len(nol2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(nol2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2[:, i].reshape(-1, 1),
                       np.array(['Weight Decay' for _ in range(len(l2))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((l2swa[:, i].reshape(-1, 1),
                       np.array(['Weight Decay w/SWA' for _ in range(len(l2swa))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(l2swa))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
    df = pd.concat((df, pd.DataFrame(np.hstack((vdp[:, i].reshape(-1, 1),
                       np.array(['VDP' for _ in range(len(vdp))]).reshape(-1, 1),
                       np.array([str(widths[i]) for _ in range(len(vdp))]).reshape(-1, 1))),
                      columns=['Spearman Correlation', 'Type', 'Layer Width'])), ignore_index=True)
df['Spearman Correlation'] = pd.to_numeric(df['Spearman Correlation'])
sns.barplot(x='Layer Width', y='Spearman Correlation', hue='Type', data=df, ci=95)
plt.title('Iris Dataset')
plt.show()