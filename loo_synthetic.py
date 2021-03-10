import os
import shap
import torch
import numpy as np
import simple_influence
from scipy.stats import spearmanr
from eli5.sklearn import PermutationImportance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize


os.environ['PYTHONHASHSEED'] = str(1234567890)


class shallow_model(torch.nn.Module):
    def __init__(self, n_feats, n_nodes, n_classes):
        super(shallow_model, self).__init__()
        self.lin1 = torch.nn.Linear(n_feats, n_nodes)
        self.lin2 = torch.nn.Linear(n_nodes, n_classes)
        self.selu = torch.nn.SELU()

    def forward(self, x):
        x = self.selu(self.lin1(x))
        x = self.lin2(x)
        return x

    def score(self, x, y):
        logits = torch.nn.functional.softmax(self.forward(x), dim=1)
        score = torch.sum(torch.argmax(logits, dim=1) == y)
        return score.numpy()


def train_net(dataset, nodes):
    x_train, x_test, y_train, y_test = dataset()
    scaler = StandardScaler()
    accs = list()
    n_epochs = 5
    device = 'cpu'
    for i in range(x_train.shape[1]):
        scaler = StandardScaler()
        x_train_loo = np.delete(x_train, i, axis=1)
        x_test_loo = np.delete(x_test, i, axis=1)
        x_train_loo_scaled = scaler.fit_transform(x_train_loo)
        x_test_loo_scaled = scaler.transform(x_test_loo)
        classifier = shallow_model(x_train.shape[1], nodes, len(np.unique(y_train))).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
        for _ in range(n_epochs):
            logits = classifier(torch.from_numpy(x_train_loo_scaled).float().to(device))
            loss = criterion(logits, torch.from_numpy(y_train).long().to(device))
            loss.backward()
            optimizer.step()
            classifier.zero_grad()
        accs.append(classifier.score(torch.from_numpy(x_test_loo_scaled).float().to(device), torch.from_numpy(y_test).long().to(device)))
        print('{}/{}'.format(i+1, x_train.shape[1]))
    return np.hstack(accs), classifier, (x_train, x_test, y_train, y_test)



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


def loo_logreg(dataset):
    x_train, x_test, y_train, y_test = dataset()
    scaler = StandardScaler()
    classifier = LogisticRegression(C=1).fit(scaler.fit_transform(x_train), y_train)
    accs = list()
    for i in range(x_train.shape[1]):
        scaler = StandardScaler()
        x_train_loo = np.delete(x_train, i, axis=1)
        x_test_loo = np.delete(x_test, i, axis=1)
        x_train_loo_scaled = scaler.fit_transform(x_train_loo)
        x_test_loo_scaled = scaler.transform(x_test_loo)
        clf = LogisticRegression().fit(x_train_loo_scaled, y_train)
        accs.append(clf.score(x_test_loo_scaled, y_test))
        print('{}/{}'.format(i+1, x_train.shape[1]))
    return np.hstack(accs), classifier, (x_train, x_test, y_train, y_test)


def influence_approx(dataset, classifier):
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    eqn_5 = simple_influence.i_pert_loss(x_train_scaled, y_train, x_test_scaled, y_test, classifier.coef_)
    return eqn_5


def kernel_shap(dataset, classifier):
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    explainer = shap.KernelExplainer(classifier.predict_proba, x_test_scaled, link="logit")
    shap_values = explainer.shap_values(x_test_scaled, nsamples=100)
    return shap_values


def permutation_importance(dataset, classifier):
    x_train, x_test, y_train, y_test = dataset
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    perm = PermutationImportance(classifier).fit(x_test_scaled, y_test)
    return perm.feature_importances_


def print_correlations(truth, observations):
    for i in range(len(observations)):
        print(spearmanr(truth, observations[i]))


def main():
    truth, classifier, dataset = loo_logreg(gen_data)
    influences = normalize(influence_approx(dataset, classifier).reshape(-1, 1))
    shap_values = np.mean(np.mean(np.dstack(kernel_shap(dataset, classifier)), axis=2), axis=0).squeeze()
    permutation = permutation_importance(dataset, classifier)
    print_correlations(truth, (influences, shap_values, permutation))


if __name__ == '__main__':
    main()
