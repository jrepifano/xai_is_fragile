import torch
import numpy as np


class influence_wrapper:
    def __init__(self, model, x_train, y_train, x_test=None, y_test=None, trainloader=None, gpu='0'):
        self.x_train = x_train
        self.y_train = y_train
        self.trainloader = trainloader
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.device = 'cuda:'+gpu if next(self.model.parameters()).is_cuda else 'cpu'

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
        for itr, (x_train, y_train) in enumerate(self.trainloader):
            self.x_train = x_train
            self.y_train = y_train
            H_i += torch.autograd.functional.hessian(self.get_train_loss, weights, vectorize=True)
        H = H_i / len(self.x_train)
        square_size = int(np.sqrt(torch.numel(H)))
        H = H.view(square_size, square_size)
        return H

    def LiSSA(self, v, weights):
        damping = 0.01
        scale = 10
        ihvp = None
        prev_norm = 1
        count = 0
        cur_estimate = v.clone()
        diff = prev_norm
        trainloader = iter(self.model.train_dataloader())
        num_samples = 60000
        while diff > 0.001 and count < 10000:
            try:
                self.x_train, self.y_train = next(trainloader)
            except StopIteration:
                trainloader = iter(self.model.train_dataloader())
            hvp = torch.autograd.functional.hvp(self.get_train_loss, weights, cur_estimate)[1]
            cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
            cur_estimate = torch.squeeze(torch.stack(cur_estimate))
            numpy_est = cur_estimate.detach().cpu().numpy()
            numpy_est = numpy_est.reshape(1, -1)
            count += 1
            diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
            prev_norm = np.linalg.norm(np.concatenate(numpy_est))
            # if count % 500 == 0:
            # print('Recursion Depth {}; Norm: {:.2f}'.format(count, prev_norm))
        if ihvp is None:
            ihvp = [b/scale for b in cur_estimate]
        else:
            ihvp = [a + b/scale for (a, b) in zip(ihvp, cur_estimate)]
            print(np.linalg.norm(np.concatenate(ihvp)))
        ihvp = torch.squeeze(torch.stack(ihvp))
        ihvp = [a / num_samples for a in ihvp]
        ihvp = torch.squeeze(torch.stack(ihvp))
        return ihvp.detach()

    def i_up_loss(self, weights, estimate=False):
        i_up_loss = list()
        test_grad = torch.autograd.grad(self.get_test_loss(weights), weights)[0]
        if estimate:
            ihvp = self.LiSSA(test_grad, weights)
            for itr, (x_train, y_train) in enumerate(self.trainloader):
                self.x_train = x_train
                self.y_train = y_train
                for i in range(len(self.x_train)):
                    self.pointer = i
                    train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                    i_up_loss.append((ihvp.view(1, -1) @ train_grad.view(-1, 1)).item())
        else:
            H = self.get_hessian(weights)
            H = H + (0.001 * torch.eye(H.shape[0], device=self.device))
            H_inv = torch.inverse(H)
            for itr, (x_train, y_train) in enumerate(self.trainloader):
                self.x_train = x_train
                self.y_train = y_train
                for i in range(len(self.x_train)):
                    self.pointer = i
                    train_grad = torch.autograd.grad(self.get_loss(weights), weights)[0]
                    i_up_loss.append((test_grad.view(1, -1) @ (H_inv @ train_grad.float().view(-1, 1))).item())
        return i_up_loss
