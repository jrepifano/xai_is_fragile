import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from lenet_swa import lenet, train, finetune, get_influence


def main():
    est_loss_diffs = list()
    true_loss_diffs = list()
    for i in range(50):
        gpu = '1'
        batch_size = 1024
        train(gpu, batch_size=batch_size)
        model = lenet(batch_size=batch_size)
        model.load_state_dict(torch.load('lenet_swa.pt'))
        model.eval()
        test_losses = model.get_losses(set='test')
        max_loss = np.argsort(test_losses)[-1]
        # max_loss = np.argsort(test_losses)[len(test_losses) // 2] # median test point
        true_loss = test_losses[max_loss]
        i_up_loss = get_influence(max_loss, batch_size=batch_size, gpu=gpu)
        top_40 = np.argsort(np.abs(i_up_loss))[::-1][:50]
        est_loss_diffs.append(i_up_loss[top_40])
        # np.savetxt('est_loss_diffs.csv', est_loss_diffs, delimiter=',')
        true_loss_diffs.append(finetune(gpu, top_40, max_loss, true_loss, batch_size=batch_size))
        # np.savetxt('true_loss_diffs.csv', true_loss_diffs, delimiter=',')
        # break
        np.save('est_loss_diffs_max_swa.npy', est_loss_diffs, allow_pickle=True)
        np.save('true_loss_diffs_max_swa.npy', true_loss_diffs, allow_pickle=True)
        print('{}/{}'.format(i, 50))


if __name__ == '__main__':
    main()
