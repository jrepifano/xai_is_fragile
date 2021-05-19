import torch
import numpy as np
from pyhessian import hessian
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class shallow_model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(shallow_model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        self.lin_last = torch.nn.Linear(n_nodes, n_classes)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(self.device)
        x = x.to(self.device)
        x = self.selu(self.lin1(x))
        x = self.lin_last(x)
        return x

    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        logits = torch.nn.functional.softmax(self.forward(x), dim=1)
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(x)
        return score.cpu().numpy()

    def fit(self, x, y):
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).long().to(self.device)
        for epoch in range(200):
            self.optimizer.zero_grad()
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            # if epoch % 10 == 0:
            #     print(loss.item())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss.item())
        train_acc = self.score(x, y)
        return train_acc.item()

class influence_wrapper():
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def get_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        # logits = self.x_train[self.pointer].view(1, -1) @ weights.T
        logits = self.model(self.x_train[self.pointer].view(1, -1))
        loss = criterion(logits, torch.tensor([self.y_train[self.pointer]], device=self.model.device))
        return loss

    def get_test_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        # logits = self.x_test.view(1, -1) @ weights.T
        logits = self.model(self.x_test.view(1, -1))
        loss = criterion(logits, torch.tensor([self.y_test], device=self.model.device))
        return loss

    def get_hessian(self, weights):
        H_i = torch.empty((3, 8, 3, 8))
        for i in range(len(self.x_train)):
            self.pointer = i
            H_i += torch.autograd.functional.hessian(self.get_loss, weights, vectorize=True)
        H = H_i / len(self.x_train)
        square_size = int(np.sqrt(torch.numel(H)))
        H = H.view(square_size, square_size)
        return H

    def i_up_params(self, weights):
        H = self.get_hessian(weights)
        H_inv = torch.inverse(H)
        i_up_params = list()
        for i in range(len(self.x_train)):
            self.pointer = i
            grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
            orig_shape = grad.shape
            i_up_params.append(-1 / len(self.x_train) * (H_inv @ grad.float().view(-1, 1)).view(orig_shape))
        return i_up_params

    def i_up_loss(self, weights):
        H = self.get_hessian(weights)
        H_inv = torch.inverse(H)
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(weights), weights)[0]
        for i in range(len(self.x_train)):
            self.pointer = i
            train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
            orig_shape = train_grad.shape
            i_up_loss.append((-test_grad.view(1, -1) @ (H_inv @ train_grad.float().view(-1, 1))).item())
        return i_up_loss


def figure_1_a_b():
    x, y = load_iris(return_X_y=True)
    x = StandardScaler().fit_transform(x)
    clf = LogisticRegression(C=1).fit(x, y)
    orig_weights = torch.tensor(clf.coef_, requires_grad=True)
    x_tensor = torch.tensor(x, requires_grad=True)
    c = influence_wrapper(x_tensor, y)
    i_up_params = c.i_up_params(orig_weights)
    est_diff = [torch.norm(i).item() for i in i_up_params]
    loo = LeaveOneOut()
    actual_difference = list()
    orig_weights = clf.coef_
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        clf = LogisticRegression(C=1).fit(x_train, y_train)
        weights = clf.coef_
        actual_difference.append(np.linalg.norm(weights - orig_weights, ord='fro'))
    est_diff = normalize(np.asarray(est_diff).reshape(1, -1))
    actual_difference = normalize(np.asarray(actual_difference).reshape(1, -1))
    print(pearsonr(actual_difference.tolist()[0], est_diff.tolist()[0]))
    plt.scatter(actual_difference, est_diff)
    plt.plot(actual_difference.tolist()[0], actual_difference.tolist()[0], 'r')
    plt.xlabel('Norm of Exact Parameter Change')
    plt.ylabel('Norm of Approximate Parameter Change')
    plt.show()
    pass


def figure_1_c_d(n_iters, n_nodes, n_layers):
    corrs, outer_train_acc, outer_test_acc, top_eig = list(), list(), list(), list()
    for n in range(n_iters):
        ###################################################################
        # Train Classif to find test instance with highest loss
        x, y = load_iris(return_X_y=True)
        x = StandardScaler().fit_transform(x)
        criterion = torch.nn.CrossEntropyLoss()
        loo = LeaveOneOut()
        train_acc, test_loss, y_pred, y_true = list(), list(), list(), list()
        for train_index, test_index in loo.split(x):
            model = shallow_model(x.shape[1], n_nodes, 3)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
            y_test = torch.from_numpy(y_test).to(model.device)
            train_acc.append(model.fit(x_train, y_train))
            logits = model(x_test)
            y_pred.append(torch.argmax(logits).item())
            y_true.append(y_test.item())
            test_loss.append(criterion(logits, y_test).item())
        max_loss = np.argmax(test_loss)
        true_test_loss = test_loss[max_loss]
        #######################################################################
        # Find training top 16.6% training points with highest loss
        train_index = np.hstack((np.arange(max_loss), np.arange(max_loss+1, len(x))))
        test_index = np.asarray([max_loss])
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loo_model = shallow_model(x.shape[1], n_nodes, 3)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        train_acc = loo_model.fit(x_train, y_train)
        y_train = torch.from_numpy(y_train).to(loo_model.device)
        hessian_comp = hessian(loo_model, torch.nn.CrossEntropyLoss(), data=(x_train, y_train), cuda=False)
        top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues()
        top_eig.append(top_eigenvalues[-1])
        train_loss = criterion(loo_model(x_train), y_train)
        top_16 = np.argsort(train_loss.cpu().detach().numpy())[::-1][:24]
        ##############################################################################
        # Train classifiers with training points removed to compute difference in loss on test point
        actual_diff = list()
        for i in top_16:
            train_index = np.hstack((np.arange(max_loss), np.arange(max_loss + 1, len(x))))
            test_index = np.asarray([max_loss])
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            model = shallow_model(x.shape[1], n_nodes, 3)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            x_train, y_train = np.delete(x_train, i, 0), np.delete(y_train, i, 0)
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
            train_acc = model.fit(x_train, y_train)
            y_test = torch.from_numpy(y_test).to(model.device)
            # x_test, y_test = [torch.from_numpy(x) for x in [x_test, y_test]]
            actual_diff.append(true_test_loss - criterion(model(x_test), y_test).item())
        ##################################################################################
        # Use Influence functions to estimate the difference in loss on test point with exact Hessian Computation
        new_x_train = scaler.transform(x[top_16])
        new_y_train = y[top_16]
        new_x_train, new_y_train = [torch.from_numpy(x) for x in [new_x_train, new_y_train]]
        loo_model.to(loo_model.device)
        c = influence_wrapper(loo_model, new_x_train.float(), new_y_train, torch.from_numpy(x_test).float(), y_test)
        i_up_loss = c.i_up_loss(loo_model.lin_last.weight)
        ##################################################################################
        # Compute spearman correlation on ranks of differences
        corr = spearmanr(actual_diff, i_up_loss).correlation
        corrs.append(corr)
        outer_train_acc.append(np.mean(train_acc))
        outer_test_acc.append(accuracy_score(y_true, y_pred))
    return corrs, outer_train_acc, outer_test_acc, top_eig


def main():
    # figure_1_a_b()
    corr, outer_train_acc, outer_test_acc, top_eig = figure_1_c_d(10, 8, 1)
    pass


if __name__ == '__main__':
    main()
