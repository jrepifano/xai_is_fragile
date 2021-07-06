import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from lenet import lenet, train, finetune, get_influence


def main():
    train()
    model = lenet(batch_size=1024)
    model.load_state_dict(torch.load('lenet.pt'))
    model.eval()
    test_losses = model.get_losses(set='test')
    max_loss = np.argsort(test_losses)[-1]
    true_loss = test_losses[max_loss]
    i_up_loss = get_influence(max_loss)
    top_40 = np.argsort(np.abs(i_up_loss))[::-1][:40]
    est_loss_diffs = i_up_loss[top_40]
    np.savetxt('est_loss_diffs.csv', est_loss_diffs, delimiter=',')
    true_loss_diffs = finetune(top_40, max_loss, true_loss)
    np.savetxt('true_loss_diffs.csv', true_loss_diffs, delimiter=',')
    print(pearsonr(true_loss_diffs, est_loss_diffs))
    print(spearmanr(true_loss_diffs, est_loss_diffs))


if __name__ == '__main__':
    main()
