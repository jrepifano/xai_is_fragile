import numpy as np
import matplotlib.pyplot as plt

approx_loss_diff = np.load('est_loss_diffs_128.npy')[6]
exact_loss_diff = np.load('true_loss_diffs_128.npy')[6]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
exact = ax1.plot(np.arange(len(approx_loss_diff)), exact_loss_diff, label='Exact Loss Differences', linewidth=2)
approx = ax2.plot(np.arange(len(approx_loss_diff)), approx_loss_diff, 'tab:orange', label='Approximate Loss Differences',linewidth=2)
ax1.set_xlabel('Training Instance Loss Rank')
ax1.set_ylabel('Loss Differences')
lns = exact + approx
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.show()