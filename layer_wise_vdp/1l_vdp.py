import os
import vdp
import time
import math
import torch
import numpy as np
from pyhessian import hessian
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def scale_hyperp(log_det, nll, kl):
    # Find the alpha scaling factor
    lli_power = orderOfMagnitude(nll)
    ldi_power = orderOfMagnitude(log_det)
    alpha = 10**(lli_power - ldi_power - 1)     # log_det_i needs to be 1 less power than log_likelihood_i

    beta = list()
    # Find scaling factor for each kl term
    kl = [i.item() for i in kl]
    smallest_power = orderOfMagnitude(np.min(kl))
    for i in range(len(kl)):
        power = orderOfMagnitude(kl[i])
        power = smallest_power-power
        beta.append(10.0**power)

    # Find the tau scaling factor
    tau = 10**(smallest_power - lli_power - 1)

    return alpha, beta, tau


class Model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(Model, self).__init__()
        self.lin1 = vdp.Linear(n_feats, n_nodes, input_flag=True)
        self.lin_last = vdp.Linear(n_nodes, n_classes)
        self.softmax = vdp.Softmax()
        self.relu = vdp.SELU()
        self.scale = False
        self.alpha = 0.1
        self.beta = [1 for layer in self.children() if hasattr(layer, 'kl_term')]
        self.tau = 0.01

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        mu, sigma = self.lin1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma

    def bottleneck(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        mu, sigma = self.lin1(x)
        mu, sigma = self.relu(mu, sigma)
        return mu, sigma

    def fit(self, x, y, no_epochs=1000):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=False)
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            mu, sigma = self.forward(x)
            log_det, nll = vdp.ELBOLoss(mu, sigma, y)
            kl = vdp.gather_kl(self)
            if not self.scale:
                self.alpha, self.beta, self.tau = scale_hyperp(log_det, nll, kl)
                self.scale = True
            loss = self.alpha * log_det + nll + self.tau * torch.stack([a * b for a, b in zip(self.beta, kl)]).sum()
            if epoch % 10000 == 0:
                print('Epoch {}/{}, Loss: {:.2f}, Train Acc: {:.2f}'.format(epoch+1, no_epochs, loss.item(), self.score(x, y)))
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        logits, sigma = self.forward(x)
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(x)
        return score.cpu().numpy()

    def get_indiv_loss(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        mu, sigma = self.forward(x)
        loss = criterion(mu, y)
        return [l.item() for l in loss] if len(loss) > 1 else loss.item()


class influence_wrapper:
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.device = 'cuda:0' if next(self.model.parameters()).is_cuda else 'cpu'

    def get_loss(self, mu, sigma):
        mu_x, sigma_x = self.model.bottleneck(self.x_train[self.pointer].reshape(1, -1))
        mu_y = mu_x @ mu.T
        sigma_y = (vdp.softplus(sigma) @ sigma_x.T).T + \
                  (mu ** 2 @ sigma_x.T).T + \
                  (mu_x ** 2 @ vdp.softplus(sigma).T)
        mu_y = torch.nn.functional.softmax(mu_y, dim=1)
        J = mu_y*(1-mu_y)
        sigma_y = (J**2) * sigma_y
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor([self.y_train[self.pointer]], device=self.device), self.model)
        return loss

    def get_train_loss(self, mu, sigma):
        mu_x, sigma_x = self.model.bottleneck(self.x_train)
        mu_y = mu_x @ mu.T
        sigma_y = (vdp.softplus(sigma) @ sigma_x.T).T + \
                  (mu ** 2 @ sigma_x.T).T + \
                  (mu_x ** 2 @ vdp.softplus(sigma).T)
        mu_y = torch.nn.functional.softmax(mu_y, dim=1)
        J = mu_y*(1-mu_y)
        sigma_y = (J**2) * sigma_y
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor(self.y_train, device=self.device), self.model)
        return loss

    def get_test_loss(self, mu, sigma):
        mu_x, sigma_x = self.model.bottleneck(self.x_test.reshape(1, -1))
        mu_y = mu_x @ mu.T + self.model.lin_last.mu.bias
        sigma_y = (vdp.softplus(sigma) @ sigma_x.T).T + \
                  (mu ** 2 @ sigma_x.T).T + \
                  (mu_x ** 2 @ vdp.softplus(sigma).T) + self.model.lin_last.sigma.bias
        mu_y = torch.nn.functional.softmax(mu_y, dim=1)
        J = mu_y*(1-mu_y)
        sigma_y = (J**2) * sigma_y
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor(self.y_test, device=self.device), self.model)
        return loss

    def get_hessian(self, mu, sigma):
        dim_1, dim_2 = mu.shape[0], mu.shape[1]
        H_i = torch.zeros((dim_1, dim_2, dim_1, dim_2), device=self.device)
        for i in range(len(self.x_train)):
            self.pointer = i
            H_i += torch.autograd.functional.hessian(self.get_loss, (mu, sigma), vectorize=True)[0][0]
        H = H_i / len(self.x_train)
        square_size = int(np.sqrt(torch.numel(H)))
        H = H.view(square_size, square_size)
        return H

    def LiSSA(self, v, weights):
        count = 0
        cur_estimate = v
        damping = 0
        scale = 10
        num_samples = len(self.x_train)
        prev_norm = 1
        diff = prev_norm
        ihvp = None
        for i in range(len(self.x_train)):
            self.pointer = i
            while diff > 0.00001 and count < 10000:
                hvp = torch.autograd.functional.hvp(self.get_train_loss, weights, cur_estimate)[1]
                cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
                cur_estimate = torch.squeeze(torch.stack(cur_estimate))  # .view(1, -1)
                numpy_est = cur_estimate.detach().cpu().numpy()
                numpy_est = numpy_est.reshape(1, -1)
                count += 1
                diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
                prev_norm = np.linalg.norm(np.concatenate(numpy_est))
            if ihvp is None:
                ihvp = [b/scale for b in cur_estimate]
            else:
                ihvp = [a + b/scale for (a, b) in zip(ihvp, cur_estimate)]
        ihvp = torch.squeeze(torch.stack(ihvp))
        ihvp = [a / num_samples for a in ihvp]
        ihvp = torch.squeeze(torch.stack(ihvp))
        return ihvp.detach()

    def i_up_loss(self, mu, sigma, idx, estimate=False):
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(mu, sigma), mu)[0]
        if estimate:
            for i in idx:
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(mu, sigma), mu)[0]
                i_up_loss.append((test_grad.view(1, -1) @ self.LiSSA(torch.autograd.functional.hvp(self.get_train_loss,
                                                                                       (mu, sigma), (train_grad))[1], mu).view(-1, 1)).detach().cpu().numpy()[0][0])
        else:
            H = self.get_hessian(mu, sigma)
            if torch.det(H) == 0:
                H = H + (0.001 * torch.eye(H.shape[0], device=self.device))
            H_inv = torch.inverse(H)
            for i in idx:
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(mu, sigma), mu)[0]
                i_up_loss.append((test_grad.view(1, -1) @ (H_inv @ train_grad.float().view(-1, 1))).item())
        return i_up_loss


def get_hessian_info(model, x, y):
    device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
    if not torch.is_tensor(x):
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
    criterion = vdp.ELBOLoss_2
    hessian_comp = hessian(model, criterion, data=(x, y), cuda=True)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    return top_eigenvalues[-1]


def find_max_loss():
    x, y = load_iris(return_X_y=True)
    loo = LeaveOneOut()
    train_acc, test_loss, y_pred = list(), list(), list()
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        model = Model(x.shape[1], 8, 3).to('cuda:0')
        model.fit(x_train, y_train, 1000)
        train_acc.append(model.score(x_train, y_train))
        test_loss.append(model.get_indiv_loss(x_test, y_test))
        logits, sigma = model(x_test)
        y_pred.append(torch.argmax(logits, dim=1).item())
    train_acc = np.mean(train_acc)
    test_acc = accuracy_score(y, y_pred)
    max_loss = np.argmax(test_loss)
    return max_loss, train_acc, test_acc


def find_top_train(max_loss=83):
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    model = Model(x.shape[1], 5, 3).to('cuda:0')
    model.fit(x_train, y_train, 60000)
    train_acc = model.score(x_train, y_train)
    train_loss = model.get_indiv_loss(x_train, y_train)
    to_look = int(1/6 * len(x-1))
    top_train = np.argsort(train_loss)[::-1][:to_look]
    top_eig = get_hessian_info(model, x_train, y_train)
    torch.save(model.state_dict(), 'loo_params_1l.pt')
    return top_train, model, top_eig, train_acc


def exact_difference(model, top_train, max_loss):
    exact_parameter_diff = list()
    exact_loss_diff = list()
    true_parameters = model.lin_last.mu.weight.detach().cpu().numpy()
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    true_loss = model.get_indiv_loss(x_test, y_test)
    for i in top_train:
        x, y = load_iris(return_X_y=True)
        train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
        test_index = np.asarray([max_loss])
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        x_train, y_train = np.delete(x_train, i, 0), np.delete(y_train, i, 0)
        # model = Model(x.shape[1], 5, 3).to('cuda:0')
        model.load_state_dict(torch.load('loo_params_1l.pt'))
        model.fit(x_train, y_train, 7500)
        # exact_parameter_diff.append(model.lin_last.weight.detach().cpu().numpy() - true_parameters)
        exact_loss_diff.append(model.get_indiv_loss(x_test, y_test) - true_loss)
    # exact_parameter_diff = [np.linalg.norm(diff, ord=2) for diff in exact_parameter_diff]
    return exact_loss_diff#, exact_parameter_diff


def approx_difference(model, top_train, max_loss):
    model.load_state_dict(torch.load('loo_params_1l.pt'))
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    infl = influence_wrapper(model, x_train, y_train, x_test, y_test)
    # approx_parameter_diff = -1 / len(x_train) * np.asarray(infl.i_up_params(model.lin_last.weight, top_train, estimate=True))
    # approx_parameter_diff = [np.linalg.norm(diff, ord=2) for diff in approx_parameter_diff]
    approx_loss_diff = np.asarray(infl.i_up_loss(model.lin_last.mu.weight, model.lin_last.sigma.weight, top_train, estimate=False))
    return approx_loss_diff#, approx_parameter_diff


def main(n_iters):
    train, eig, pearson, spearman = list(), list(), list(), list()
    i = 0
    while i < n_iters:
        try:
            start_time = time.time()
            # max_loss, train_acc, test_acc = find_max_loss()  # 83 is always the highest loss then 133, 70, 77
            max_loss = 83
            # print('Done max loss')
            top_train, model, top_eig, train_acc = find_top_train(max_loss)
            print(train_acc)
            print('Done top train')
            exact_loss_diff = exact_difference(model, top_train, max_loss)
            print('Done Exact Diff')
            approx_loss_diff = approx_difference(model, top_train, max_loss)
            p = pearsonr(exact_loss_diff, approx_loss_diff)
            s = spearmanr(exact_loss_diff, approx_loss_diff)
        except Exception:
            continue
        train.append(train_acc)
        eig.append(top_eig)
        pearson.append(p[0])
        spearman.append(s[0])
        print('Done {}/{} in {:.2f} minutes'.format(i+1, n_iters, (time.time()-start_time)/60))
        if i % 5 == 0:
            np.save('figure1/vdp_1l_train.npy', train, allow_pickle=True)
            np.save('figure1/vdp_1l_eig.npy', eig, allow_pickle=True)
            np.save('figure1/vdp_1l_pearson.npy', pearson, allow_pickle=True)
            np.save('figure1/vdp_1l_spearman.npy', spearman, allow_pickle=True)
        i += 1
    np.save('figure1/vdp_1l_train.npy', train, allow_pickle=True)
    np.save('figure1/vdp_1l_eig.npy', eig, allow_pickle=True)
    np.save('figure1/vdp_1l_pearson.npy', pearson, allow_pickle=True)
    np.save('figure1/vdp_1l_spearman.npy', spearman, allow_pickle=True)


if __name__ == '__main__':
    main(n_iters=50)
