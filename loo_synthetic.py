import os
import shap
import numpy as np
import simple_influence
from scipy.stats import spearmanr
from eli5.sklearn import PermutationImportance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

os.environ['PYTHONHASHSEED'] = str(1234567890)


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
    # flip sign so it correlates with accuracy in the right directions
    shap_values = np.mean(np.mean(np.dstack(kernel_shap(dataset, classifier)), axis=2), axis=0).squeeze()
    permutation = permutation_importance(dataset, classifier)
    print_correlations(truth, (influences, shap_values, permutation))


if __name__ == '__main__':
    main()
