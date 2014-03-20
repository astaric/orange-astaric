from collections import namedtuple

import os
import numpy as np
import scipy.stats as stats
import sklearn
import sklearn.mixture
import Orange
from Orange.data import Domain, Table
from Orange.data.imputation import ImputeTable
from Orange.feature import Continuous
from sklearn.cluster import _k_means
from sklearn.cluster.k_means_ import _squared_norms

from orangecontrib.astaric.gmm import em


def continuous_uci_datasets():
    datasets_dir = os.path.join(os.path.dirname(Orange.__file__), 'datasets')
    for ds in [file for file in os.listdir(datasets_dir) if file.endswith('tab')]:
        if ds in ["adult_sample.tab", "horse-colic_learn.tab", "horse-colic_test.tab"]:
            continue
        table = Table(ds)
        continuous_features = [a for a in table.domain.features if isinstance(a, Continuous)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), ImputeTable(table))
            new_table.name = ds
            yield new_table


def GDS_datasets():
    datasets_dir = '/Users/anze/Downloads'
    for ds in [file for file in os.listdir(datasets_dir) if file.startswith('GDS') and file.endswith('tab')]:
        table = Table(os.path.join(datasets_dir, ds))
        continuous_features = [a for a in table.domain.features if isinstance(a, Continuous)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), ImputeTable(table))
            new_table.name = ds
            yield new_table


Result = namedtuple('LAC', ['means', 'covars'])


def KM(X, k):
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    Y = kmeans.fit_predict(X)

    means = kmeans.cluster_centers_
    covars = np.zeros((k, X.shape[1]))
    for j in range(k):
        xn = X[Y == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / len(xn)
    return Result(means, covars)


def LAC(X, k):
    w, means, covars, priors = em(X, k, 100)
    squared_norms = _squared_norms(X)
    labels = - np.ones(X.shape[0], np.int32)
    distances = np.zeros(shape=(0,), dtype=np.float64)
    _k_means._assign_labels_array(X, squared_norms, means, labels, distances=distances)
    covars = np.zeros((k, X.shape[1]))
    for j in range(k):
        xn = X[labels == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / (len(xn) if len(xn) else 1.)
    return Result(means, covars)


def GMM(X, k):
    gmm = sklearn.mixture.GMM(n_components=k)
    gmm.fit(X)
    means = gmm.means_
    squared_norms = _squared_norms(X)
    labels = - np.ones(X.shape[0], np.int32)
    distances = np.zeros(shape=(0,), dtype=np.float64)
    _k_means._assign_labels_array(X, squared_norms, means, labels, distances=distances)
    covars = np.zeros((k, X.shape[1]))
    for j in range(k):
        xn = X[labels == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / (len(xn) if len(xn) else 1.)
    return Result(means, covars)



def score(result, x):
    means = x.mean(axis=0)
    xn = x - means
    stdev = np.sqrt(np.sum(xn ** 2, axis=0) / len(xn))

    return np.sqrt(result.covars / stdev).sum() / xn.shape[1]

print r"\begin{tabular}{ l r r r }"
print r"dataset & S(k-means) & S(gmm)& S(lac) \\"
print r"\hline"
# for ds in [Table('iris')]:
# for ds in GDS_datasets():
results = []
for ds in continuous_uci_datasets():
    x, = ds.to_numpy("a")
    x_ma, = ds.to_numpy_MA("a")
    means = x_ma.mean(axis=0)
    col_mean = stats.nanmean(x, axis=0)
    inds = np.where(x_ma.mask)
    x[inds] = np.take(col_mean, inds[1])

    k = 10
    n_steps = 99

    km = KM(x, k)
    lac = LAC(x, k)
    gmm = GMM(x, k)
    km_score, gmm_score, lac_score = map(lambda r: score(r, x), [km, gmm, lac])
    results.append((km_score, gmm_score, lac_score))
    print r"%s & %.2f & %.2f & %.2f \\" % (ds.name.replace("_", "\_"), km_score, gmm_score, lac_score)
print r"\end{tabular}"

import pylab as plt
results = np.array(results)
km = results[1:, 0]
gmm = results[1:, 1]
lac = results[1:, 2]

plt.plot(gmm, lac, 'x')
plt.plot([0, 80], [0, 80])
plt.xlabel("gmm")
plt.ylabel("lac")
plt.show()

#ranks = np.array([[1, 2] if s[0] < s[1] else [2, 1] for s in results])
#print ranks[0]
#from Orange.evaluation.scoring import graph_ranks
#graph_ranks('ranks.png', ranks, ["kmeans", 'lac'])



