import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from pyhessian import hessian
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.lin_last = torch.nn.Linear(84, 10)

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        if x.device.type != device:
            x = x.to(device)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.lin_last(x)
        return x


    def bottleneck(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        if x.device.type != device:
            x = x.to(device)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def fit(self, x, y, no_epochs=1000):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        dataloader = DataLoader(data_loader(x, y), 10000, shuffle=True, num_workers=2, pin_memory=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)
        for epoch in range(no_epochs):
            for idx, (x, y) in enumerate(dataloader):
                if x.device.type != device or y.device.type != device:
                    x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self.forward(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step(loss.item())
            if scheduler.optimizer.param_groups[0]['lr'] == 1.0000000000000004e-08:
                # print('Early Stopping Triggered')
                break
            # if epoch % 10 == 0:
            #     print('Epoch {}/{}, Loss: {:.2f}, Train Acc: {:.2f}'.format(epoch+1, no_epochs, loss.item(), self.score(x, y)))


    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        if x.device.type != device or y.device.type != device:
            x, y = x.to(device), y.to(device)
        logits = torch.nn.functional.softmax(self.forward(x), dim=1)
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(x)
        return score.cpu().numpy()

    def get_indiv_loss(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        if x.device.type != device or y.device.type != device:
            x, y = x.to(device), y.to(device)
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
        logits = self.model.bottleneck(self.x_train[self.pointer].unsqueeze(0))
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
        logits = self.model.bottleneck(self.x_test)
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
        damping = 0.01
        scale = 10
        num_samples = len(self.x_train)
        ihvp = None
        for i in range(len(self.x_train)):
            self.pointer = i
            prev_norm = 1
            diff = prev_norm
            while diff > 0.00001 and count < 10000:
                hvp = torch.autograd.functional.hvp(self.get_loss, weights, cur_estimate)[1]
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
            H = H + (0.001 * torch.eye(H.shape[0], device=self.device))
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


def train_model():
    mnist_train = MNIST('/data/', train=True, download=False)
    mnist_test = MNIST('/data/', train=False, download=False)
    x_train, y_train = mnist_train.data.view(60000, 1, 28, 28)/255.0, mnist_train.targets
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28)/255.0, mnist_test.targets
    model = Model().to('cuda:0')
    model.fit(x_train, y_train, 5000)
    train_acc = model.score(x_train, y_train)
    test_loss = model.get_indiv_loss(x_test, y_test)
    test_acc = model.score(x_test, y_test)
    max_loss = np.argmax(test_loss)
    med_loss = np.argsort(test_loss)[len(test_loss)//2]
    train_loss = model.get_indiv_loss(x_train, y_train)
    to_look = 40
    top_train = np.argsort(train_loss)[::-1][:to_look]
    torch.save(model.state_dict(), 'lenet.pt')
    return model, (med_loss, max_loss), top_train, (train_acc, test_acc)


def approx_difference(model, test_idx, top_train):
    med_loss = test_idx[0]
    max_loss = test_idx[1]
    to_look = 40
    model.load_state_dict(torch.load('lenet.pt'))
    mnist_train = MNIST('/data/', train=True, download=False)
    mnist_test = MNIST('/data/', train=False, download=False)
    x_train, y_train = mnist_train.data.view(60000, 1, 28, 28)/255.0, mnist_train.targets
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28)/255.0, mnist_test.targets
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28) / 255.0, mnist_test.targets
    test_index_max = np.asarray([max_loss])
    test_index_med = np.asarray([med_loss])
    x_test_max, y_test_max = x_test[test_index_max], y_test[test_index_max]
    x_test_med, y_test_med = x_test[test_index_med], y_test[test_index_med]

    # infl = influence_wrapper(model, x_train, y_train, x_test_med, y_test_med)
    # approx_loss_diff_med = np.asarray(infl.i_up_loss(model.lin_last.weight, estimate=False))

    infl = influence_wrapper(model, x_train, y_train, x_test_max, y_test_max)
    approx_loss_diff_max = np.asarray(infl.i_up_loss(model.lin_last.weight, estimate=False))
    top_train_max = np.argsort(np.abs(approx_loss_diff_max))[::-1][:to_look]
    return approx_loss_diff_max[top_train_max], top_train_max#approx_loss_diff_max[top_train]


def exact_difference(model, top_train_max, test_idx):
    med_loss = test_idx[0]
    max_loss = test_idx[1]
    exact_loss_diff_med, exact_loss_diff_max = list(), list()
    mnist_test = MNIST('/data/', train=False, download=False)
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28)/255.0, mnist_test.targets
    test_index = np.asarray([med_loss])
    true_loss_med = model.get_indiv_loss(x_test[test_index], y_test[test_index])
    test_index = np.asarray([max_loss])
    true_loss_max = model.get_indiv_loss(x_test[test_index], y_test[test_index])
    for i in top_train_max:
        mnist_train = MNIST('/data/', train=True, download=False)
        mnist_test = MNIST('/data/', train=False, download=False)
        x_train, y_train = mnist_train.data.view(60000, 1, 28, 28) / 255.0, mnist_train.targets
        x_test, y_test = mnist_test.data.view(10000, 1, 28, 28) / 255.0, mnist_test.targets
        test_index_max = np.asarray([max_loss])
        # test_index_med = np.asarray([med_loss])
        x_test_max, y_test_max = x_test[test_index_max], y_test[test_index_max]
        # x_test_med, y_test_med = x_test[test_index_med], y_test[test_index_med]
        x_train, y_train = np.delete(x_train, i, 0), np.delete(y_train, i, 0)
        model.load_state_dict(torch.load('lenet.pt'))
        model.fit(x_train, y_train, 3000)
        # exact_loss_diff_med.append(model.get_indiv_loss(x_test_med, y_test_med) - true_loss_med)
        exact_loss_diff_max.append(model.get_indiv_loss(x_test_max, y_test_max) - true_loss_max)
    return exact_loss_diff_max


def main():
    train, test, pearson, spearman = list(), list(), list(), list()
    for i in range(5):
        start_time = time.time()
        model, (med_loss, max_loss), top_train, (train_acc, test_acc) = train_model()
        print('Done training')
        approx_loss_diff_max, top_train_max = approx_difference(model, (med_loss, max_loss), top_train)
        print('Done approx')
        exact_loss_diff_max = exact_difference(model, top_train_max, (med_loss, max_loss))
        print('Done Exact Diff')

        train.append(train_acc)
        test.append(test_acc)

        max_pearson = pearsonr(exact_loss_diff_max, approx_loss_diff_max)
        max_spearman = spearmanr(exact_loss_diff_max, approx_loss_diff_max)
        # med_pearson = pearsonr(exact_loss_diff_med, approx_loss_diff_med)
        # med_spearman = spearmanr(exact_loss_diff_med, approx_loss_diff_med)
        print('Done {}/{} in {:.2f} minutes'.format(i+1, 10, (time.time()-start_time)/60))
        print(train_acc)
        print(test_acc)
        print(max_pearson)
        print(max_spearman)
        # print(med_pearson)
        # print(med_spearman)
        pass


if __name__ == '__main__':
    main()
