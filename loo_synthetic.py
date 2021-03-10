import os
import shap
import torch
import numpy as np
import simple_influence
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from eli5.permutation_importance import get_score_importances


os.environ['PYTHONHASHSEED'] = str(1234567890)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gen_data():
    n_samples = np.random.randint(100, 10000)
    print('Number of Samples in DS: ' + str(n_samples))
    n_feats = np.random.choice([10, 20, 50, 100], 1).item()
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()
    X, y = make_classification(n_samples, n_feats, n_feats // 2, 0, 0, 2, n_clusters, None, 0, sep, True, 0, 1, hyper, shuffle=False)
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.2)
    return X, x_test, y, y_test


class shallow_model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(shallow_model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        self.lin_last = torch.nn.Linear(n_nodes, n_classes)
        self.selu = torch.nn.SELU()

    def forward(self, x):
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


def train_net(dataset, nodes):
    x_train, x_test, y_train, y_test = dataset
    accs = list()
    n_epochs = 10
    device = 'cuda:0'
    scaler = StandardScaler()
    x_train_loo_scaled = scaler.fit_transform(x_train)
    x_test_loo_scaled = scaler.transform(x_test)
    classifier_all_feats = shallow_model(x_train.shape[1], nodes, len(np.unique(y_train))).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_all_feats.parameters(), lr=1e-2, weight_decay=0.001)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = classifier_all_feats(torch.from_numpy(x_train_loo_scaled).float().to(device))
        loss = criterion(logits, torch.from_numpy(y_train).long().to(device))
        loss.backward()
        optimizer.step()
    for i in range(x_train.shape[1]):
        scaler = StandardScaler()
        x_train_loo = np.delete(x_train, i, axis=1)
        x_test_loo = np.delete(x_test, i, axis=1)
        x_train_loo_scaled = scaler.fit_transform(x_train_loo)
        x_test_loo_scaled = scaler.transform(x_test_loo)
        classifier = shallow_model(x_train_loo.shape[1], nodes, len(np.unique(y_train))).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=0.001)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = classifier(torch.from_numpy(x_train_loo_scaled).float().to(device))
            loss = criterion(logits, torch.from_numpy(y_train).long().to(device))
            loss.backward()
            optimizer.step()
        accs.append(classifier.score(torch.from_numpy(x_test_loo_scaled).float().to(device), torch.from_numpy(y_test).long().to(device)))
        # print('{}/{}'.format(i+1, x_train.shape[1]))
    return np.hstack(accs), classifier_all_feats, (x_train, x_test, y_train, y_test)


def influence_approx(dataset, classifier):
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    eqn_5 = simple_influence.i_pert_loss(x_train_scaled, y_train, x_test_scaled, y_test, classifier)
    return eqn_5


def gradient_shap(dataset, classifier):
    device = 'cuda:0'
    classifier.to(device)
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = torch.from_numpy(scaler.fit_transform(x_train)).float().to(device)
    x_test_scaled = torch.from_numpy(scaler.transform(x_test)).float().to(device)
    explainer = shap.GradientExplainer(classifier, x_train_scaled, local_smoothing=0.2)
    shap_values = explainer.shap_values(x_test_scaled)
    return shap_values


def permutation_importance(dataset, classifier):
    device = 'cuda:0'
    classifier.to(device)
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    base_score, score_decreases = get_score_importances(classifier.score, x_test_scaled, y_test)
    perm_importances = np.mean(score_decreases, axis=0)
    return perm_importances


def get_correlation(truth, observations):
    corr = list()
    pvalue = list()
    for i in range(len(observations)):
        spear = spearmanr(truth, observations[i])
        corr.append(spear.correlation)
        pvalue.append(spear.pvalue)
    return corr, pvalue


def main():
    n_datasets = 10000
    nodes = [5, 10, 15, 20, 50, 100, 1000]
    corr_results = np.empty((n_datasets, len(nodes), 3))
    pvalue_results = np.empty((n_datasets, len(nodes), 3))
    for i in range(n_datasets):
        dataset = gen_data()
        for j in range(len(nodes)):
            truth, classifier, dataset = train_net(dataset, nodes[j])
            influences = normalize(influence_approx(dataset, classifier).reshape(1, -1))[0]
            shap_values = np.mean(np.mean(np.dstack(gradient_shap(dataset, classifier)), axis=2), axis=0).squeeze()
            permutation = permutation_importance(dataset, classifier)
            corr, pvalue = get_correlation(truth, (influences, shap_values, permutation))
            corr_results[i, j, :] = corr
            pvalue_results[i, j, :] = pvalue
        if i % 100 == 0:
            print('{}/{}'.format(i+1, n_datasets))
            np.save(os.getcwd()+'/results/corr_width_{}.npy'.format(i), corr_results)
            np.save(os.getcwd()+'/results/pvalue_width_{}.npy'.format(i), pvalue_results)


if __name__ == '__main__':
    main()
