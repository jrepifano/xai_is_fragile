import os
import time
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
np.random.seed(6)
torch.manual_seed(14)


class Model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        # self.lin2 = torch.nn.Linear(n_nodes, n_nodes)
        self.lin_last = torch.nn.Linear(n_nodes, n_classes)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        x = self.relu(self.lin1(x))
        # x = self.relu(self.lin2(x))
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        x = self.relu(self.lin1(x))
        # x = self.relu(self.lin2(x))
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
        logits = logits @ weights.T
        loss = criterion(logits, torch.tensor([self.y_train[self.pointer]], device=self.device))
        return loss

    def get_train_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_train)
        logits = logits @ weights.T
        loss = criterion(logits, torch.tensor(self.y_train, device=self.device))
        return loss

    def get_test_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_test.reshape(1, -1))
        logits = logits @ weights.T
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
        return H.detach().cpu()

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

    def i_up_loss(self, weights, estimate=False):
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(weights), weights)[0]
        if estimate:
            ihvp = self.LiSSA(test_grad, weights)
            for i in range(len(self.x_train)):
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                i_up_loss.append((-ihvp.view(1, -1) @ train_grad.view(-1, 1)).item())
        else:
            H = self.get_hessian(weights)
            H_inv = torch.inverse(H)
            for i in range(len(self.x_train)):
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
        model = Model(x.shape[1], 5, 3).to('cuda:0')
        model.fit(x_train, y_train)
        train_acc.append(model.score(x_train, y_train))
        test_loss.append(model.get_indiv_loss(x_test, y_test))
        y_pred.append(torch.argmax(torch.nn.functional.softmax(model(x_test), dim=1)).item())
    train_acc = np.mean(train_acc)
    test_acc = accuracy_score(y, y_pred)
    max_loss = np.argmax(test_loss)
    return max_loss, train_acc, test_acc


def train_model(max_loss=83):
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    model = Model(x.shape[1], 5, 3).to('cuda:0')
    model.fit(x_train, y_train, 60000)
    top_eig = get_hessian_info(model, x_train, y_train)
    torch.save(model.state_dict(), 'loo_params.pt')
    return model, top_eig


def exact_difference(model, top_train, max_loss):
    exact_parameter_diff = list()
    exact_loss_diff = list()
    true_parameters = model.lin_last.weight.detach().cpu().numpy()
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
        model = Model(x.shape[1], 5, 3).to('cuda:0')
        model.load_state_dict(torch.load('loo_params.pt'))
        model.fit(x_train, y_train, 7500)
        # exact_parameter_diff.append(model.lin_last.weight.detach().cpu().numpy() - true_parameters)
        exact_loss_diff.append(model.get_indiv_loss(x_test, y_test) - true_loss)
    exact_parameter_diff = [np.linalg.norm(diff, ord=2) for diff in exact_parameter_diff]
    return exact_loss_diff#, exact_parameter_diff


def approx_difference(model, max_loss):
    model.load_state_dict(torch.load('loo_params.pt'))
    x, y = load_iris(return_X_y=True)
    train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
    test_index = np.asarray([max_loss])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    infl = influence_wrapper(model, x_train, y_train, x_test, y_test)
    approx_loss_diff = np.asarray(infl.i_up_loss(model.lin_last.weight, estimate=True))
    to_look = int(1/6 * len(x-1))
    top_train = np.argsort(approx_loss_diff)[::-1][:to_look]
    return approx_loss_diff[top_train], top_train


def main():
    outer_start_time = time.time()
    train, test, eig, pearson, spearman = list(), list(), list(), list(), list()
    for i in range(5):
        start_time = time.time()
        # max_loss, train_acc, test_acc = find_max_loss()  # 83 is always the highest loss then 133, 70, 77
        # print('Done max loss')
        max_loss = 83
        model, top_eig = train_model()
        print('Done training')
        approx_loss_diff, top_train = approx_difference(model, max_loss)
        print('Done approx diff')
        exact_loss_diff = exact_difference(model, top_train, max_loss)
        print('Done Exact Diff')
        # train.append(train_acc)
        # test.append(test_acc)
        eig.append(top_eig)
        pearson.append(pearsonr(exact_loss_diff, approx_loss_diff))
        spearman.append(spearmanr(exact_loss_diff, approx_loss_diff))
        print('Done {}/{} in {:.2f} minutes'.format(i+1, 10, (time.time()-start_time)/60))
        print(train)
        print(test)
        print(pearson)
        print(spearman)
        print(eig)
    # np.save('figure1/det_1l_l2_approx_train.npy', train)
    # np.save('figure1/det_1l_l2_approx_test.npy', test)
    # np.save('figure1/det_1l_l2_approx_eig.npy', eig)
    # np.save('figure1/det_1l_l2_approx_pearson.npy', pearson)
    # np.save('figure1/det_1l_l2_approx_spearman.npy', spearman)
    # print('Finished Iter in {:.2f} minutes'.format((time.time()-outer_start_time)/60))
    #     plt.plot(exact_parameter_diff, exact_parameter_diff, 'r')
    #     plt.scatter(exact_parameter_diff, approx_parameter_diff)
    #     plt.xlabel('Norm of Exact Parameter Change')
    #     plt.ylabel('Norm of Approximate Parameter Change')
        # plt.xlim(min(exact_parameter_diff), max(exact_parameter_diff))
        # plt.ylim(min(exact_parameter_diff), max(exact_parameter_diff))
        # plt.show()
        # print('Train/Test Accuracy: {:.2f}/{:.2f}'.format(train_acc, test_acc))
        # print('Pearson: {:.2f}'.format(pearson))
        # print('Spearman: {:.2f}'.format(spearman))
    pass


if __name__ == '__main__':
    main()

# Train/Test Accuracy: 0.97/0.97
# Pearson: 1.00
# Spearman: 0.86

# [0.9725726]
# [0.9666666666666667]
# [0.6113434398393797]
# [0.6176923076923077]
# [1.4167283773422241]

# [0.9732886]
# [0.9533333333333334]
# [-0.16284023723086935]
# [0.04230769230769231]
# [1.2044204473495483]