import vdp
import math
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


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
    tau = 10**(smallest_power - lli_power - 3)

    return alpha, beta, tau


class Model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(Model, self).__init__()
        self.lin1 = vdp.Linear(n_feats, n_nodes, input_flag=True)
        self.lin2 = vdp.Linear(n_nodes, n_nodes)
        self.lin3 = vdp.Linear(n_nodes, n_nodes)
        self.lin4 = vdp.Linear(n_nodes, n_nodes)
        self.lin5 = vdp.Linear(n_nodes, n_nodes)
        self.lin6 = vdp.Linear(n_nodes, n_nodes)
        self.lin7 = vdp.Linear(n_nodes, n_nodes)
        self.lin8 = vdp.Linear(n_nodes, n_nodes)
        self.lin_last = vdp.Linear(n_nodes, n_classes)
        self.softmax = vdp.Softmax()
        self.relu = vdp.SELU()
        self.alpha = 0.1
        self.beta = [1 for layer in self.children() if hasattr(layer, 'kl_term')]
        self.tau = 0.01
        self.scale = False

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        mu, sigma = self.lin1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin4(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin5(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin6(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin7(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin8(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma

    def score(self, x, y):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
        logits, sigma = self.forward(x)
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(x)
        return score.cpu().numpy()


def main():
    x, y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = Model(x.shape[1], 5, 3)
    model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
    if not torch.is_tensor(x):
        x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)
    for epoch in range(60000):
        optimizer.zero_grad()
        mu, sigma = model(x)
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(model)
        if not model.scale:
            model.alpha, model.beta, model.tau = scale_hyperp(log_det, nll, kl)
            model.scale = True
            print(model.alpha)
            print(model.beta)
            print(model.tau)
        loss = model.alpha * log_det + nll + model.tau * torch.stack([a * b for a, b in zip(model.beta, kl)]).sum()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(loss.item())
            print(model.score(x, y))


if __name__ == '__main__':
    main()
