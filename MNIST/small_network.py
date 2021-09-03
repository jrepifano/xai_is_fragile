import os
import sys
import torch
import numpy as np
from scipy.stats import spearmanr
from torchvision.datasets import MNIST


class Model(torch.nn.Module):
    def __init__(self, width=32):
        super(Model, self).__init__()
        self.width = width
        self.layer_1 = torch.nn.Linear(28 * 28, width)
        self.lin_last = torch.nn.Linear(width, 10)
        self.relu = torch.nn.ReLU()
        self.alpha = None
        self.tau = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.lin_last(x)
        return x

    def bottleneck(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        return x

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y) / len(logits)
        return score.cpu().numpy()

    def get_loss(self, x, y):
        loss = self.criterion(x, y)
        return loss


def train(model, no_epochs, train_idx_to_remove=None, save=False):
    mnist_train = MNIST(os.getcwd(), train=True, download=False)
    if train_idx_to_remove != None:
        mnist_train.data = np.delete(mnist_train.data, train_idx_to_remove, axis=0)
        mnist_train.targets = np.delete(mnist_train.targets, train_idx_to_remove, axis=0)
        x_train, y_train = mnist_train.data.view(59999, -1) / 255.0, mnist_train.targets
    else:
        x_train, y_train = mnist_train.data.view(60000, -1) / 255.0, mnist_train.targets
    mnist_test = MNIST(os.getcwd(), train=False, download=False)

    x_test, y_test = mnist_test.data.view(10000, -1)/255.0, mnist_test.targets
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=False)
    if not save:
        for param in model.parameters():
            param.requires_grad = False
        model.lin_last.weight.requires_grad = True
        model.lin_last.bias.requires_grad = True
    model.to('cuda:0')
    train_accs = []
    test_accs = []
    for epoch in range(no_epochs):
        total_loss = 0
        model.train()
        optimizer.zero_grad()
        logits = model.forward(x_train.float().to('cuda:0'))
        loss = model.get_loss(logits, y_train.to('cuda:0'))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        train_acc = model.score(logits, y_train.to('cuda:0'))
        train_accs.append(train_acc)
        model.eval()
        logits = model.forward(x_test.float().to('cuda:0'))
        test_acc = model.score(logits, y_test.to('cuda:0'))
        test_accs.append(test_acc)
        # if (epoch % 500) == 0 or (epoch == no_epochs-1):
        #     print('Epoch {}/{}: Training Loss: {:.2f}'.format(epoch + 1, no_epochs, total_loss))
        #     print('Train Accuracy: {:.2f}'.format(train_acc))
    if save:
        torch.save(model.state_dict(), 'fullycon_'+str(model.width)+'.pt')
    y_test = y_test.to('cuda:0')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    test_losses = criterion(logits, y_test)
    return [i.item() for i in test_losses]


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
        loss = criterion(logits, torch.tensor([self.y_train[self.pointer]], device=self.device).long())
        return loss

    def get_train_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_train)
        logits = logits @ weights.T + self.model.lin_last.bias
        loss = criterion(logits, torch.tensor(self.y_train, device=self.device).long())
        return loss

    def get_test_loss(self, weights):
        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model.bottleneck(self.x_test.reshape(1, -1))
        logits = logits @ weights.T + self.model.lin_last.bias
        loss = criterion(logits, torch.tensor([self.y_test], device=self.device).long())
        return loss

    def get_hessian(self, weights):
        dim_1, dim_2 = weights.shape[0], weights.shape[1]
        H_i = torch.zeros((dim_1, dim_2, dim_1, dim_2), device=self.device)
        for i in range(len(self.x_train)):
            self.pointer = i
            H_i += torch.autograd.functional.hessian(self.get_loss, weights, vectorize=True)[0][0]
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
        prev_norm = 1
        diff = prev_norm
        ihvp = None
        while diff > 0.00001 and count < 10000:
            # for i in range(len(self.x_train)):
            #     self.pointer = i
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
            # print('Recursion Depth {}; Norm: {:.2f}'.format(count, prev_norm))
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


def get_infl(model, max_loss):
    i_up_loss = list()
    mnist_train = MNIST(os.getcwd(), train=True, download=False)
    mnist_test = MNIST(os.getcwd(), train=False, download=False)
    x_train, y_train = mnist_train.data.view(60000, -1) / 255.0, mnist_train.targets
    x_test, y_test = (mnist_test.data.view(10000, -1)/255.0)[max_loss], (mnist_test.targets)[max_loss]
    infl = influence_wrapper(model, x_train, y_train, x_test, y_test)
    i_up_loss.append(infl.i_up_loss(model.lin_last.weight, estimate=True))
    i_up_loss = np.hstack(i_up_loss)
    return i_up_loss


def main(width=32, gpu=0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    est_loss_diffs = list()
    true_loss_diffs = list()
    for i in range(50):
        model = Model(width=width)
        inner_loss_diffs = list()
        test_losses = train(model, no_epochs=5000, save=True)
        model = Model(width=width)
        model.load_state_dict(torch.load('fullycon_'+str(width)+'.pt'))
        max_loss = np.argsort(test_losses)[-1]
        true_loss = test_losses[max_loss]
        i_up_loss = get_infl(model, max_loss)
        top_40 = np.argsort(np.abs(i_up_loss))[::-1][:40]
        est_loss_diffs.append(i_up_loss[top_40])
        for j in range(len(top_40)):
            model = Model(width=width)
            model.load_state_dict(torch.load('fullycon_'+str(width)+'.pt'))
            test_losses = train(model, no_epochs=2000, train_idx_to_remove=top_40[j])
            inner_loss_diffs.append(test_losses[max_loss] - true_loss)
            print('outer {}/{}, in {}/{}'.format(i+1, 50, j+1, 40))
        true_loss_diffs.append(np.hstack(inner_loss_diffs))
        print(spearmanr(true_loss_diffs[i], est_loss_diffs[i]))
        np.save('est_loss_diffs_'+str(width)+'.npy', est_loss_diffs, allow_pickle=True)
        np.save('true_loss_diffs_'+str(width)+'.npy', true_loss_diffs, allow_pickle=True)
        pass

if __name__ == '__main__':
    try:
        width = int(sys.argv[1])
        gpu = int(sys.argv[2])
        main(width, gpu)
    except Exception:
        main(width=8, gpu=0)
