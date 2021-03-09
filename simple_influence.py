import torch
import random
import numpy as np
from torch.autograd import grad


def hessian_vector_product(ys, xs, v):
    """
    Compute Hessian Vector Product
    Ensure all functions of the model are twice differentiable
    i.e. change ReLUs to SELUs
    :param ys: train loss, [tensor, with grad]
    :param xs: parameters [tensor, with grad]
    :param v: gradient of test loss, [tensor with grad]
    :return: H * gradient of test loss
    """
    J = grad(ys, xs, create_graph=True)[0]
    grads = grad(J, xs, v, retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_loss, test_loss, layer_weight):
    """
    Estimate Inverse Hessian Vector Product
    :param train_loss: train loss (some type of cross-entropy loss), [tensor, with grad]
    :param test_loss: test loss (some type of cross-entopy loss), [tensor, with grad]
    :param layer_weight: weights of the last layer of the model e.g. model.layer_2.weight - suggested by Koh, Pang 2017
    :param model: pytorch model, [pytorch model object]
    :return: H^-1 * gradient of test loss
    """
    scale = 10
    damping = 0
    num_samples = 1
    v = grad(test_loss, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001 and count < 10000:
            hvp = hessian_vector_product(train_loss, layer_weight, cur_estimate)
            # hvp = torch.autograd.functional.hvp(train_loss, layer_weight, cur_estimate)
            cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
            cur_estimate = torch.squeeze(torch.stack(cur_estimate)).view(1, -1)
            numpy_est = cur_estimate.detach().cpu().numpy()
            numpy_est = numpy_est.reshape(1, -1)

            if (count % 100 == 0):
                print("Recursion at depth %s: norm is %.8lf" % (count, np.linalg.norm(np.concatenate(numpy_est))))
            count += 1
            diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
            prev_norm = np.linalg.norm(np.concatenate(numpy_est))
            ihvp = [b / scale for b in cur_estimate]
            ihvp = torch.squeeze(torch.stack(ihvp))
            ihvp = [a / num_samples for a in ihvp]
            ihvp = torch.squeeze(torch.stack(ihvp))
    return ihvp.detach()


def i_pert_loss(x_train, y_train, x_test, y_test, coefs, device='cpu'):
    """
    Estimate the effect that upweighting the inputs of a training instance will have on the loss of a test instance
    :param x_train: training data, [n_samples, n_feats]
    :param y_train: training labels, [n_samples]
    :param x_test: test instance or test data, [n_samples, n_feats]
    :param y_test: test labels, [n_samples]
    :param model: PyTorch model object
    :param layer_weight: Weight of layer to compute influence at. Koh, Pang 2017 suggest final layer
    :param n: If you'd like to smooth influence by using an average of the neighborhood increase n, [int]
              This intuition is from Smilkov, et.al. 2017 (SmoothGrad)
    :param std: Standard deviation of noise to add to sample during smoothing
    :param criterion: Some type of Cross-entropy function
    :param device: Device to compute on. Much faster on cuda devices
    :return: gradient with respect to input (gradient of training loss of each traning instance * H^-1 * gradient of test loss)
    """
    x_train, x_test = torch.from_numpy(x_train).float().to(device), torch.from_numpy(x_test).float().to(device)
    y_train, y_test = torch.from_numpy(y_train).float().to(device), torch.from_numpy(y_test).float().to(device)
    params = torch.from_numpy(coefs).float().to(device)

    x_train.requires_grad = True
    x_test.requires_grad = True
    # y_train.requires_grad = True
    # y_test.requires_grad = True
    params.requires_grad = True

    criterion = torch.nn.BCEWithLogitsLoss()
    train_loss = criterion(x_train @ params.T, y_train.view(-1, 1))
    test_loss = criterion(x_test @ params.T, y_test.view(-1, 1))
    ihvp = lissa(train_loss, test_loss, params)

    x = x_train
    x.requires_grad = True
    x_out = x_train @ params.T
    x_loss = criterion(x_out, y_train.view(-1, 1))
    grads = grad(x_loss, params, create_graph=True)[0]
    grads = grads.squeeze()
    grads = grads.view(1, -1).squeeze()
    infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(x_train)
    i_pert = grad(infl, x, retain_graph=False)
    i_pert = i_pert[0]

    eqn_5 = -i_pert.detach().cpu().numpy()

    return np.sum(eqn_5, axis=0)
