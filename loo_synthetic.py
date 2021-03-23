import os
import shap
import torch
import numpy as np
import simple_influence
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from eli5.permutation_importance import get_score_importances


os.environ['PYTHONHASHSEED'] = str(1234567890)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gen_data():
    n_samples = np.random.randint(100, 5000)
    # n_samples = 1000
    print('Number of Samples in DS: ' + str(n_samples))
    n_feats = np.random.choice([10, 20, 50, 100], 1).item()
    n_feats = 20
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()
    X, y = make_classification(n_samples, n_features=n_feats, n_informative=n_feats // 2,
                               n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=n_clusters,
                               weights=None, flip_y=0, class_sep=sep, hypercube=hyper, shift=0, scale=1, shuffle=False)
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


def train_net(dataset, nodes, n_epochs):
    x_train, x_test, y_train, y_test = dataset
    accs = list()
    device = 'cuda:0'
    scaler = StandardScaler()
    x_train_loo_scaled = scaler.fit_transform(x_train)
    x_test_loo_scaled = scaler.transform(x_test)
    classifier_all_feats = shallow_model(x_train.shape[1], nodes, len(np.unique(y_train))).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_all_feats.parameters(), lr=1e-3, weight_decay=0.001)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = classifier_all_feats(torch.from_numpy(x_train_loo_scaled).float().to(device))
        loss = criterion(logits, torch.from_numpy(y_train).long().to(device))
        loss.backward()
        optimizer.step()
    train_acc = classifier_all_feats.score(x_train_loo_scaled, y_train)
    test_acc = classifier_all_feats.score(x_test_loo_scaled, y_test)
    for i in range(x_train.shape[1]):
        scaler = StandardScaler()
        x_train_loo = np.delete(x_train, i, axis=1)
        x_test_loo = np.delete(x_test, i, axis=1)
        x_train_loo_scaled = scaler.fit_transform(x_train_loo)
        x_test_loo_scaled = scaler.transform(x_test_loo)
        classifier = shallow_model(x_train_loo.shape[1], nodes, len(np.unique(y_train))).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=0.001)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = classifier(torch.from_numpy(x_train_loo_scaled).float().to(device))
            loss = criterion(logits, torch.from_numpy(y_train).long().to(device))
            loss.backward()
            optimizer.step()
        accs.append(classifier.score(torch.from_numpy(x_test_loo_scaled).float().to(device), torch.from_numpy(y_test).long().to(device)))
        # print('{}/{}'.format(i+1, x_train.shape[1]))
    return np.hstack(accs), classifier_all_feats, (train_acc, test_acc), (x_train, x_test, y_train, y_test)


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
    shap_values = explainer.shap_values(x_test_scaled, nsamples=100)
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


def get_accs(n_feats, observations):
    accs = list()
    inform_feats = set(range(n_feats // 2))
    for i in range(len(observations)):
        obs_feats = set(np.argsort(abs(observations[i]))[::-1][:n_feats//2])
        accs.append(len(inform_feats.intersection(obs_feats)) / (n_feats//2))
    return accs


def get_pearson(truth, test_acc, observations):
    stat = list()
    pvalue = list()
    for i in range(len(observations)):
        if i == 2:
            stat_i, pvalue_i = pearsonr(test_acc - truth, test_acc-observations[i])
        else:
            stat_i, pvalue_i = pearsonr(test_acc-truth, np.abs(observations[i]))
        stat.append(stat_i)
        pvalue.append(pvalue_i)
    return stat, pvalue


def get_spearman(truth, test_acc, observations):
    stat = list()
    pvalue = list()
    for i in range(len(observations)):
        stat_i, pvalue_i = spearmanr(np.argsort(test_acc-truth), np.argsort(np.abs(observations[i])))
        stat.append(stat_i)
        pvalue.append(pvalue_i)
    return stat, pvalue


def main():
    n_datasets = 10000
    nodes = [100, 500, 1000, 2000, 5000]
    epochs = [300, 300, 350, 350, 350]
    accuracy_results = np.empty((n_datasets, len(nodes), 5))
    spearman_stats = np.empty((n_datasets, len(nodes), 3))
    spearman_pvalues = np.empty((n_datasets, len(nodes), 3))
    pearson_stats = np.empty((n_datasets, len(nodes), 3))
    pearson_pvalues = np.empty((n_datasets, len(nodes), 3))
    for i in range(n_datasets):
        dataset = gen_data()
        for j in range(len(nodes)):
            truth, classifier, (tt_acc), dataset = train_net(dataset, nodes[j], epochs[j])
            print('Finished Truth')
            influences = normalize(influence_approx(dataset, classifier).reshape(1, -1))[0]
            print('Finished Influence')
            shap_values = np.mean(np.mean(np.dstack(gradient_shap(dataset, classifier)), axis=2), axis=0).squeeze()
            print('Finished SHAP')
            permutation = permutation_importance(dataset, classifier)
            print('Finished Permutation')
            infl_acc, shap_acc, permute_acc = get_accs(dataset[0].shape[1], (influences, shap_values, permutation))
            pearson_stat, pearson_pvalue = get_spearman(truth, tt_acc[1], (influences, shap_values, permutation))
            spearman_stat, spearman_pvalue = get_spearman(truth, tt_acc[1], (influences, shap_values, permutation))
            accuracy_results[i, j, :] = [tt_acc[0].item(), tt_acc[1].item(), infl_acc, shap_acc, permute_acc]
            spearman_stats[i, j, :] = spearman_stat
            spearman_pvalues[i, j, :] = spearman_pvalue
            pearson_stats[i, j, :] = pearson_stat
            pearson_pvalues[i, j, :] = pearson_pvalue
        print('{}/{}'.format(i, n_datasets))
        if i % 100 == 0:
            np.save(os.getcwd()+ '/results/accuracies_width_{}.npy'.format(i), accuracy_results)
            np.save(os.getcwd()+ '/results/pearson_width_{}.npy'.format(i), pearson_stats)
            np.save(os.getcwd() + '/results/pearson_pvalue_width_{}.npy'.format(i), pearson_pvalues)
            np.save(os.getcwd() + '/results/spearman_width_{}.npy'.format(i), spearman_stats)
            np.save(os.getcwd() + '/results/spearman_pvalue_width_{}.npy'.format(i), spearman_pvalues)
    np.save(os.getcwd() + '/results/accuracies_width_final.npy', accuracy_results)
    np.save(os.getcwd() + '/results/pearson_width_final.npy', pearson_stats)
    np.save(os.getcwd() + '/results/pearson_pvalue_width_final.npy', pearson_pvalues)
    np.save(os.getcwd() + '/results/spearman_width_final.npy', spearman_stats)
    np.save(os.getcwd() + '/results/spearman_pvalue_width_final.npy', spearman_pvalues)


if __name__ == '__main__':
    main()
