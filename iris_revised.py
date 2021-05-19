import torch
import numpy as np
from pyhessian import hessian
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


class Model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        self.lin_last = torch.nn.Linear(n_nodes, n_classes)
        self.selu = torch.nn.SELU()

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(device)
        x = self.selu(self.lin1(x))
        x = self.lin_last(x)
        return x

    def fit(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-2, weight_decay=0.001, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=False)
        for epoch in range(500):
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

    def get_loss(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        logits = self.forward(x)
        loss = criterion(logits, y)
        return [l.item() for l in loss]


class influence_wrapper():
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def get_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model(self.x_train[self.pointer].view(1, -1))
        loss = criterion(logits, torch.tensor([self.y_train[self.pointer]], device=self.model.device))
        return loss

    def get_train_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model(self.x_train)
        loss = criterion(logits, torch.tensor(self.y_train, device=self.model.device))
        return loss

    def get_test_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model(self.x_test.view(1, -1))
        loss = criterion(logits, torch.tensor([self.y_test], device=self.model.device))
        return loss

    def get_hessian(self, weights):
        H = 1/len(self.x_train) * torch.autograd.functional.hessian(self.get_train_loss, weights, vectorize=True)
        square_size = int(np.sqrt(torch.numel(H)))
        H = H.view(square_size, square_size)
        return H

    def i_up_params(self, weights, idx):
        H = self.get_hessian(weights)
        H_inv = torch.inverse(H)
        i_up_params = list()
        for i in idx:
            self.pointer = i
            grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
            orig_shape = grad.shape
            i_up_params.append(-1 / len(self.x_train) * (H_inv @ grad.float().view(-1, 1)).view(orig_shape))
        return i_up_params

    def i_up_loss(self, weights, idx):
        H = self.get_hessian(weights)
        H_inv = torch.inverse(H)
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(weights), weights)[0]
        for i in idx:
            self.pointer = i
            train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
            i_up_loss.append((-test_grad.view(1, -1) @ (H_inv @ train_grad.float().view(-1, 1))).item())
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
        model = Model(x.shape[1], 8, 3).to('cuda:0')
        model.fit(x_train, y_train)
        train_acc.append(model.score(x_train, y_train))
        test_loss.append(model.get_loss(x_test, y_test))
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
    model = Model(x.shape[1], 8, 3).to('cuda:0')
    model.fit(x_train, y_train)
    train_loss = model.get_loss(x_train, y_train)
    to_look = int(1/6 * len(x-1))
    top_train = np.argsort(train_loss)[::-1][:to_look]
    top_eig = get_hessian_info(model, x_train, y_train)
    return top_train, model, top_eig


def main():
    # max_loss, train_acc, test_acc = find_max_loss() # 83 is always the highest loss then 133, 70, 77
    top_train, model, top_eig = find_top_train()
    pass


if __name__ == '__main__':
    main()
