import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from lenet import lenet, train, finetune, get_influence


def main():
    est_loss_diffs = list()
    true_loss_diffs = list()
    for i in range(50):
        # train()
        model = lenet(batch_size=8192)
        model.load_state_dict(torch.load('lenet.pt'))
        model.eval()
        test_losses = model.get_losses(set='test')
        max_loss = np.argsort(test_losses)[-1]
        true_loss = test_losses[max_loss]
        i_up_loss = get_influence(max_loss)
        top_40 = np.argsort(np.abs(i_up_loss))[::-1][:10]
        est_loss_diffs.append(i_up_loss[top_40])
        # np.savetxt('est_loss_diffs.csv', est_loss_diffs, delimiter=',')
        true_loss_diffs.append(finetune(top_40, max_loss, true_loss))
        # np.savetxt('true_loss_diffs.csv', true_loss_diffs, delimiter=',')
        print(pearsonr(true_loss_diffs[0], est_loss_diffs[0]))
        print(spearmanr(true_loss_diffs[0], est_loss_diffs[0]))
        # np.save('est_loss_diffs_vdp.npy', est_loss_diffs, allow_pickle=True)
        # np.save('true_loss_diffs_vdp.npy', true_loss_diffs, allow_pickle=True)
        print('{}/{}'.format(i, 50))


if __name__ == '__main__':
    main()
