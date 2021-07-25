import os
import time
import torch
import numpy as np
from pyhessian import hessian
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# Random Seed - Negating the randomizing effect
np.random.seed(6)

# Seeds : 2, 5, 10, 13, 15, 20
# Random Seed for tensorflow
torch.manual_seed(14)


class Model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        self.lin_last = torch.nn.Linear(n_nodes, n_classes)
        self.relu = torch.nn.SELU()

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        x = self.relu(self.lin1(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        x = self.relu(self.lin1(x))
        return x

    def fit(self, x, y, no_epochs=1000):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=False)
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            logits = self.forward(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        logits = torch.nn.functional.softmax(self.forward(x), dim=1)
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(x)
        return score.cpu().numpy()

    def get_indiv_loss(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        logits = self.forward(x)
        loss = criterion(logits, y)
        return [l.item() for l in loss] if len(loss) > 1 else loss.item()


class influence_wrapper:
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.device = 'cuda:0' if next(self.model.parameters()).is_cuda else 'cpu'

    def get_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_train[self.pointer].reshape(1, -1))
        logits = logits @ weights.T + self.model.lin_last.bias
        loss = criterion(logits, torch.tensor([self.y_train[self.pointer]], device=self.device))
        return loss

    def get_train_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_train)
        logits = logits @ weights.T + self.model.lin_last.bias
        loss = criterion(logits, torch.tensor(self.y_train, device=self.device))
        return loss

    def get_test_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_test.reshape(1, -1))
        logits = logits @ weights.T + self.model.lin_last.bias
        loss = criterion(logits, torch.tensor(self.y_test, device=self.device))
        return loss

    def get_hessian(self, weights):
        dim_1, dim_2 = weights.shape[0], weights.shape[1]
        H_i = torch.zeros((dim_1, dim_2, dim_1, dim_2), device=self.device)
        for i in range(len(self.x_train)):
            self.pointer = i
            H_i += torch.autograd.functional.hessian(self.get_loss, weights, vectorize=True)
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

    def i_up_params(self, weights, idx, estimate=False):
        i_up_params = list()
        if estimate:
            for i in idx:
                self.pointer = i
                grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                i_up_params.append(self.LiSSA(torch.autograd.functional.hvp(self.get_train_loss, weights, grad)[1], weights).detach().cpu().numpy())
        else:
            H = self.get_hessian(self.model.lin_last.weight)
            H_inv = torch.inverse(H)
            for i in idx:
                self.pointer = i
                grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                orig_shape = grad.shape
                i_up_params.append((H_inv @ grad.float().view(-1, 1)).view(orig_shape).detach().cpu().numpy())
        return i_up_params

    def i_up_loss(self, weights, idx, estimate=False):
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(weights), weights)[0]
        if estimate:
            for i in idx:
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                i_up_loss.append((test_grad.view(1, -1) @ self.LiSSA(torch.autograd.functional.hvp(self.get_train_loss,
                                                                                       weights, train_grad)[1], weights).view(-1, 1)).detach().cpu().numpy()[0][0])
        else:
            H = self.get_hessian(weights)
            H_inv = torch.inverse(H)
            for i in idx:
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                i_up_loss.append((test_grad.view(1, -1) @ (H_inv @ train_grad.float().view(-1, 1))).item())
        return i_up_loss


def get_hessian_info(model, x, y):
    device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
    if not torch.is_tensor(x):
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
    criterion = torch.nn.CrossEntropyLoss()
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
        model = Model(x.shape[1], 20, 3).to('cuda:0')
        model.fit(x_train, y_train)
        train_acc.append(model.score(x_train, y_train))
        test_loss.append(model.get_indiv_loss(x_test, y_test))
        y_pred.append(torch.argmax(torch.nn.functional.softmax(model(x_test), dim=1)).item())
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
    model = Model(x.shape[1], 20, 3).to('cuda:0')
    model.fit(x_train, y_train, 60000)
    train_acc = model.score(x_train, y_train)
    train_loss = model.get_indiv_loss(x_train, y_train)
    to_look = int(1/6 * len(x-1))
    top_train = np.argsort(train_loss)[::-1][:to_look]
    top_eig = get_hessian_info(model, x_train, y_train)
    torch.save(model.state_dict(), 'loo_params_20w.pt')
    return top_train, model, top_eig, train_acc


def exact_difference(model, top_train, max_loss):
    exact_loss_diff = list()
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
        model = Model(x.shape[1], 20, 3).to('cuda:0')
        model.load_state_dict(torch.load('loo_params_20w.pt'))
        model.fit(x_train, y_train, 7500)
        exact_loss_diff.append(model.get_indiv_loss(x_test, y_test) - true_loss)
    return exact_loss_diff


def approx_difference(model, top_train, max_loss):
    model.load_state_dict(torch.load('loo_params_20w.pt'))
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    infl = influence_wrapper(model, x_train, y_train, x_test, y_test)
    approx_loss_diff = np.asarray(infl.i_up_loss(model.lin_last.weight, top_train, estimate=False))
    return approx_loss_diff


def main():
    outer_start_time = time.time()
    train, eig, pearson, spearman = list(), list(), list(), list()
    for i in range(1):
        start_time = time.time()
        # max_loss, train_acc, test_acc = find_max_loss()  # 83 is always the highest loss then 133, 70, 77
        # print('Done max loss')
        max_loss = 83
        top_train, model, top_eig, train_acc = find_top_train(max_loss)
        print('Done top train')
        exact_loss_diff = exact_difference(model, top_train, max_loss)
        print('Done Exact Diff')
        approx_loss_diff = approx_difference(model, top_train, max_loss)
        train.append(train_acc)
        eig.append(top_eig)
        pearson.append(pearsonr(exact_loss_diff, approx_loss_diff)[0])
        spearman.append(spearmanr(exact_loss_diff, approx_loss_diff)[0])
        print('Done {}/{} in {:.2f} minutes'.format(i+1, 10, (time.time()-start_time)/60))
        if i % 10 == 0:
            np.save('figure1/det_20w_l2_train.npy', train)
            np.save('figure1/det_20w_l2_eig.npy', eig)
            np.save('figure1/det_20w_l2_pearson.npy', pearson)
            np.save('figure1/det_20w_l2_spearman.npy', spearman)
    np.save('figure1/det_20w_l2_train.npy', train)
    np.save('figure1/det_20w_l2_eig.npy', eig)
    np.save('figure1/det_20w_l2_pearson.npy', pearson)
    np.save('figure1/det_20w_l2_spearman.npy', spearman)
    print('Finished Iter in {:.2f} minutes'.format((time.time()-outer_start_time)/60))

    pass


if __name__ == '__main__':
    main()
