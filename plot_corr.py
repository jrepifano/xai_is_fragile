import numpy as np
import matplotlib.pyplot as plt

nodes = [5, 10, 15, 20, 50, 100, 1000]
corr = np.load('results/corr_width_900.npy')
avg_corr = np.nanmean(np.abs(corr[:900]), axis=0)
print(avg_corr)
plt.plot(nodes, avg_corr[:, 0], label='influence')
plt.plot(nodes, avg_corr[:, 1], label='SHAP')
plt.plot(nodes, avg_corr[:, 2], label='Permutation')
plt.legend()
plt.xlabel('Number of hidden nodes')
plt.ylabel('Average Spearman Correlation')
plt.legend()
plt.show()
