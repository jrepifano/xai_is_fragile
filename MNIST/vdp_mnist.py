import os
import vdp
import time
import math
import torch
import numpy as np
import torch.nn.functional as F
from pyhessian import hessian
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def scale_hyperp(log_det, nll, kl):
    # Find the alpha scaling factor
    lli_power = orderOfMagnitude(nll)
    ldi_power = orderOfMagnitude(log_det)
    alpha = 10**(lli_power - ldi_power - 2)     # log_det_i needs to be 1 less power than log_likelihood_i

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
        self.conv1 = vdp.Conv2d(1, 6, 5, input_flag=True)
        self.conv2 = vdp.Conv2d(6, 16, 5)
        self.pool = vdp.MaxPool2d(2, 2)
        self.fc1 = vdp.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = vdp.Linear(120, 84)
        self.lin_last = vdp.Linear(84, 10)
        self.relu = vdp.ReLU()
        self.softmax = vdp.Softmax()
        self.scale = False
        self.alpha = 0.1
        self.beta = [1 for layer in self.children() if hasattr(layer, 'kl_term')]
        self.tau = 0.01

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        if x.device.type != device:
            x = x.to(device)
        mu, sigma = self.conv1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        mu, sigma = torch.flatten(mu, 1), torch.flatten(sigma, 1)
        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.fc2(mu, sigma)
        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma


    def bottleneck(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        if x.device.type != device:
            x = x.to(device)
        mu, sigma = self.conv1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)
        mu, sigma = torch.flatten(mu, 1), torch.flatten(sigma, 1)
        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.fc2(mu, sigma)
        return mu, sigma

    def fit(self, x, y, no_epochs=1000):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        dataloader = DataLoader(data_loader(x, y), 100, shuffle=True, num_workers=2, pin_memory=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        for epoch in range(no_epochs):
            for idx, (x, y) in enumerate(dataloader):
                if x.device.type != device or y.device.type != device:
                    x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                mu, sigma = self.forward(x)
                log_det, nll = vdp.ELBOLoss(mu, sigma, y)
                kl = vdp.gather_kl(self)
                if not self.scale:
                    self.alpha, self.beta, self.tau = scale_hyperp(log_det, nll, kl)
                    self.scale = True
                loss = self.alpha * log_det + nll + self.tau * torch.stack([a * b for a, b in zip(self.beta, kl)]).sum()
                loss.backward()
                optimizer.step()
            scheduler.step(loss.item())
            if scheduler.optimizer.param_groups[0]['lr'] == 1.0000000000000004e-08:
                print('Early Stopping Triggered')
                break
            if epoch % 10 == 0:
                print('Epoch {}/{}, Loss: {:.2f}, Train Acc: {:.2f}'.format(epoch+1, no_epochs, loss.item(), self.score(x, y)))


    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        dataloader = DataLoader(data_loader(x, y), 100, shuffle=True, num_workers=2, pin_memory=True)
        score = 0
        for idx, (x, y) in enumerate(dataloader):
            if x.device.type != device or y.device.type != device:
                x, y = x.to(device), y.to(device)
            logits, sigma = self.forward(x)
            score += torch.sum(torch.argmax(logits, dim=1) == y)
        score = score/len(x)
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
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor([self.y_train[self.pointer]], device=self.device))#, self.model)
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
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor(self.y_train, device=self.device))#, self.model)
        return loss

    def get_test_loss(self, mu, sigma):
        mu_x, sigma_x = self.model.bottleneck(self.x_test.reshape(1, -1))
        mu_y = mu_x @ mu.T
        sigma_y = (vdp.softplus(sigma) @ sigma_x.T).T + \
                  (mu ** 2 @ sigma_x.T).T + \
                  (mu_x ** 2 @ vdp.softplus(sigma).T)
        mu_y = torch.nn.functional.softmax(mu_y, dim=1)
        J = mu_y*(1-mu_y)
        sigma_y = (J**2) * sigma_y
        loss = vdp.ELBOLoss_2((mu_y, sigma_y), torch.tensor(self.y_test, device=self.device))#, self.model)
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

    def LiSSA(self, v, mu, sigma):
        count = 0
        cur_estimate = v
        damping = 0
        scale = 10
        num_samples = len(self.x_train)
        ihvp = None
        prev_norm = 1
        diff = prev_norm
        for i in range(len(self.x_train)):
            self.pointer = i
            while diff > 0.00001 and count < 10000:
                hvp = torch.autograd.functional.hvp(self.get_train_loss, (mu, sigma), (cur_estimate, torch.zeros_like(cur_estimate)))[1][0]
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

    def i_up_loss(self, mu, sigma, estimate=False):
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(mu, sigma), mu)[0]
        if estimate:
            ihvp = self.LiSSA(test_grad, mu, sigma)
            for i in range(len(self.x_train)):
                self.pointer = i
                train_grad = torch.autograd.grad(self.get_loss(mu, sigma), mu)[0]
                i_up_loss.append((-ihvp.view(1, -1) @ train_grad.view(-1, 1)).item())
        else:
            H = self.get_hessian(mu, sigma)
            # H = H + (0.001 * torch.eye(H.shape[0], device=self.device))
            H_inv = torch.inverse(H)
            for i in range(len(self.x_train)):
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


def train_model():
    mnist_train = MNIST(os.getcwd(), train=True, download=False)
    mnist_test = MNIST(os.getcwd(), train=False, download=False)
    x_train, y_train = mnist_train.data.view(60000, 1, 28, 28)/255.0, mnist_train.targets
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28)/255.0, mnist_test.targets
    model = Model().to('cuda:0')
    model.fit(x_train, y_train, 1)
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
    mnist_train = MNIST(os.getcwd(), train=True, download=False)
    mnist_test = MNIST(os.getcwd(), train=False, download=False)
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
    approx_loss_diff_max = np.asarray(infl.i_up_loss(model.lin_last.mu.weight, model.lin_last.sigma.weight, estimate=False))
    top_train_max = np.argsort(np.abs(approx_loss_diff_max))[::-1][:to_look]
    return approx_loss_diff_max[top_train_max], top_train_max#approx_loss_diff_max[top_train]


def exact_difference(model, top_train_max, test_idx):
    med_loss = test_idx[0]
    max_loss = test_idx[1]
    exact_loss_diff_med, exact_loss_diff_max = list(), list()
    mnist_test = MNIST(os.getcwd(), train=False, download=False)
    x_test, y_test = mnist_test.data.view(10000, 1, 28, 28)/255.0, mnist_test.targets
    test_index = np.asarray([med_loss])
    true_loss_med = model.get_indiv_loss(x_test[test_index], y_test[test_index])
    test_index = np.asarray([max_loss])
    true_loss_max = model.get_indiv_loss(x_test[test_index], y_test[test_index])
    for i in top_train_max:
        mnist_train = MNIST(os.getcwd(), train=True, download=False)
        mnist_test = MNIST(os.getcwd(), train=False, download=False)
        x_train, y_train = mnist_train.data.view(60000, 1, 28, 28) / 255.0, mnist_train.targets
        x_test, y_test = mnist_test.data.view(10000, 1, 28, 28) / 255.0, mnist_test.targets
        test_index_max = np.asarray([max_loss])
        # test_index_med = np.asarray([med_loss])
        x_test_max, y_test_max = x_test[test_index_max], y_test[test_index_max]
        # x_test_med, y_test_med = x_test[test_index_med], y_test[test_index_med]
        x_train, y_train = np.delete(x_train, i, 0), np.delete(y_train, i, 0)
        model.load_state_dict(torch.load('lenet.pt'))
        model.fit(x_train, y_train, 1)
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
