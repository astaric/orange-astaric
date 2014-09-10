from collections import namedtuple
from Orange.evaluation.scoring import compute_CD
from math import sqrt
from itertools import chain

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

np.random.seed(42)


def iris():
    yield Table("iris")

def continuous_uci_datasets():
    datasets_dir = os.path.join(os.path.dirname(Orange.__file__), 'datasets')
    for ds in [file for file in os.listdir(datasets_dir) if file.endswith('tab')]:
        if ds in ["adult.tab"]:
            continue
        if ds in ["adult_sample.tab", "horse-colic_learn.tab", "horse-colic_test.tab"]:
            continue
        table = Table(ds)
        continuous_features = [a for a in table.domain.features if isinstance(a, Continuous)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), ImputeTable(table))
            new_table.name = ds
            yield new_table


def GDS_datasets():
    datasets_dir = '/Users/anze/dev/orange-astaric/orangecontrib/astaric/GDS'
    for ds in [file for file in os.listdir(datasets_dir) if file.startswith('GDS') and file.endswith('tab')]:
        table = Table(os.path.join(datasets_dir, ds))
        continuous_features = [a for a in table.domain.features if isinstance(a, Continuous)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), ImputeTable(table))
            new_table.name = ds
            yield new_table


def temporal_datasets():
    datasets_dir = '/Users/anze/dev/orange-astaric/orangecontrib/astaric/temporal'
    for ds in [file for file in os.listdir(datasets_dir) if file.endswith('tab')]:
        table = Table(os.path.join(datasets_dir, ds))
        continuous_features = [a for a in table.domain.features if isinstance(a, Continuous)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), ImputeTable(table))
            new_table.name = ds
            yield new_table


Result = namedtuple('results', ['means', 'covars', 'k', 'minis'])


def KM(X, k):
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    Y = kmeans.fit_predict(X)

    means = kmeans.cluster_centers_
    covars = np.zeros((k, X.shape[1]))
    for j in range(k):
        xn = X[Y == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / len(xn)
    return Result(means, covars, k, [])


def LAC(X, k):
    w, means, covars, priors = em(X, k, 100)
    squared_norms = _squared_norms(X)
    labels = - np.ones(X.shape[0], np.int32)
    distances = np.zeros(shape=(0,), dtype=np.float64)
    _k_means._assign_labels_array(X, squared_norms, means, labels, distances=distances)
    covars = np.zeros((k, X.shape[1]))

    realk = 0
    realmeans = []
    realcovars = []
    for j in range(k):
        xn = X[labels == j, :] - means[j]
        if len(xn):
            realk += 1
            realmeans.append(means[j])
            realcovars.append(np.sum(xn ** 2, axis=0) / len(xn))
    return Result(np.array(realmeans), np.array(realcovars), realk, [])


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
    return Result(means, covars, k, [])


def parallel_coordinates_plot(filename, X, means=None, stdevs=None, annotate=lambda ax: None):
    import pylab as plt
    from matplotlib.patches import Polygon
    colors = "#a6cee3,#1f78b4,#b2df8a,#33a02c,#fb9a99,#e31a1c,#fdbf6f,#ff7f00,#cab2d6,#6a3d9a".split(",")

    fig, ax = plt.subplots()
    for y in X:
        x = np.array(range(len(y)))
        ax.plot(x, y, color='k', alpha=.01)

    for i in range(X.shape[1]):
        ax.plot([i, i], [0, 1], color='k', lw=1.)

    if means is not None and stdevs is not None:
        for i, (mean, stdev) in enumerate(zip(means, stdevs)):
            y = [mean[j] + stdev[j] for j in range(len(mean))]
            y += [mean[j] - stdev[j] for j in range(len(mean))][::-1]
            x = range(len(mean)) + range(len(mean))[::-1]

            poly = Polygon(zip(x, y), facecolor=colors[i], edgecolor='none', alpha=.5)
            plt.gca().add_patch(poly)

            y = [mean[j] + stdev[j] / 2 for j in range(len(mean))]
            y += [mean[j] - stdev[j] / 2 for j in range(len(mean))][::-1]
            x = range(len(mean)) + range(len(mean))[::-1]

            poly = Polygon(zip(x, y), facecolor=colors[i], edgecolor='none', alpha=.8)
            plt.gca().add_patch(poly)

    ax.set_xlim([-.01, X.shape[1] - 0.93])
    annotate(ax)
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight')


def covariance_score(result, x):
    means = x.mean(axis=0)
    xn = x - means
    stdev = np.sqrt(np.sum(xn ** 2, axis=0) / len(xn))

    return np.sqrt(result.covars / stdev).sum() / xn.shape[1]


def global_probability_score(result, x):
    score = 0.

    dims = range(x.shape[1])
    w = np.empty((result.k, len(x)))
    for j in range(result.k):
        det = result.covars[j, dims].prod()
        inv_covars = 1. / result.covars[j, dims]
        xn = x[:, dims] - result.means[j, dims]
        factor = (2.0 * np.pi) ** (len(dims) / 2.0) * det ** 0.5
        w[j] = np.exp(-.5 * np.sum(xn * inv_covars * xn, axis=1)) / factor
    w /= w.sum(axis=0)
    w = np.nan_to_num(w)

    for i in range(result.k):
        for j in range(result.k):
            if i == j:
                continue
            score += (w[j, :] * w[i, :]).sum() / len(x)
    return score


def score_dimension_pair(x, result, dims):
    w = np.empty((result.k, len(x)))
    for j in range(result.k):
        det = result.covars[j, dims].prod()
        inv_covars = 1. / result.covars[j, dims]
        xn = x[:, dims] - result.means[j, dims]
        factor = (2.0 * np.pi) ** (len(dims) / 2.0) * det ** 0.5
        w[j] = np.exp(-.5 * np.sum(xn * inv_covars * xn, axis=1)) / factor
    w = np.nan_to_num(w)
    w /= w.sum(axis=0)
    w = np.nan_to_num(w)

    return sum(
        (w[j, :] * w[i, :]).sum() / len(x)
        for i in range(result.k)
        for j in range(result.k)
        if i != j
    )


def probability_score(result, x):
    return sum(
        score_dimension_pair(x, result, dims)
        for dims in zip(range(x.shape[1]), range(1, x.shape[1])))


def best_triplet_variance_score(result, x):
    score = 0.
    for j in range(result.k):
        minvar = None
        mini = None
        for i in range(1, x.shape[1] - 1):
            var = np.sum(result.covars[j, i-1:i+2])
            if minvar is None or var < minvar:
                minvar = var
                mini = i
        score += minvar
        result.minis.append(mini)
    score /= result.k
    return score


def best_triplet_probability_score(result, x):
    return min(
        score_dimension_pair(x, result, dims)
        for dims in zip(range(x.shape[1]), range(1, x.shape[1]), range(2, x.shape[1])))


def reorder_features_by_similarity(sim_matrix):
    order = []
    for i in range(sim_matrix.shape[0]):
        sim_matrix[i, i] = float('-inf')
    maxvaridx = sim_matrix.argmax()

    def use(i):
        sim_matrix[i, :] = float('-inf')
        sim_matrix[:, i] = float('-inf')

    order.append(maxvaridx % len(sim_matrix))
    while len(order) < len(sim_matrix):
        a = order[-1]
        b = order[0]
        amaxidx = sim_matrix[a, :].argmax()
        amax = sim_matrix[a, amaxidx]
        bmaxidx = sim_matrix[b, :].argmax()
        bmax = sim_matrix[b, bmaxidx]

        if amax > bmax:
            order.append(amaxidx)
            use(a)
        else:
            order.insert(0, bmaxidx)
            use(b)
    return order


def reorder_attributes_covariance(x, k=0):
    cov = np.cov(x, rowvar=0)
    #cov = np.abs(cov)
    return x[:, reorder_features_by_similarity(cov)]


def reorder_attributes_probability_score(x, k):
    m = x.shape[1]
    gmm = GMM(x, k)
    probability = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i >= j:
                continue
            probability[i, j] = probability[j, i] = \
                score_dimension_pair(x, gmm, (i, j))
    return x[:, reorder_features_by_similarity(probability)]



def test(datasets=(),
         normalization="stdev", reorder='none', score='probability',
         print_latex=True):
    global results
    #print "%% normalization=%s,reorder=%s,score=%s, %%" % (normalization, reorder, score)

    if print_latex:
        print r"\begin{tabular}{ l r r r }"
        print r"dataset & S(k-means) & S(gmm)& S(lac) \\"
        print r"\hline"
    results = []
    #for ds in [Table('vehicle', name='vehicle')]:
    #for ds in GDS_datasets():
    for ds in datasets:
    #for ds in temporal_datasets():
        x, = ds.to_numpy("a")
        x_ma, = ds.to_numpy_MA("a")
        means = x_ma.mean(axis=0)
        col_mean = stats.nanmean(x, axis=0)
        inds = np.where(x_ma.mask)
        x[inds] = np.take(col_mean, inds[1])

        k = 10
        n_steps = 99

        if normalization == 'none':
            pass
        elif normalization == '01':
            m, M = x.min(axis=0), x.max(axis=0)
            x = (x - m) / (M - m)
        elif normalization == 'stdev':
            xn = x - means
            stdev = np.sqrt(np.sum(xn ** 2, axis=0) / len(xn))
            x /= stdev
        else:
            raise AttributeError('Unknonwn normalization type')

        if reorder == 'none':
            pass
        elif reorder == 'shuffle':
            np.random.shuffle(x.T)
        elif reorder == 'covariance':
            x = reorder_attributes_covariance(x)
        elif reorder == 'probability':
            x = reorder_attributes_probability_score(x, k)
        else:
            raise AttributeError('Unknown feature reordering type')

        if score == 'covariance':
            scorer = covariance_score
        elif score == 'probability':
            scorer = probability_score
        elif score == 'global_probability':
            scorer = global_probability_score
        elif score == 'best_triplet':
            scorer = best_triplet_variance_score
        elif score == 'best_triplet_probability':
            scorer = best_triplet_probability_score

        lac = LAC(x, k)
        km = KM(x, lac.k)
        gmm = GMM(x, lac.k)

        if lac.k < 2:
            continue

        km_score, gmm_score, lac_score = map(lambda r: scorer(r, x), [km, gmm, lac])
        results.append((km_score, gmm_score, lac_score))
        if not print_latex:
            print "%s,%s,%s,%s,%f,%f,%f,%i" % (normalization, reorder, score, ds.name, km_score, gmm_score, lac_score, lac.k)
            continue
        print "%s & " % ds.name.replace("_", "\_"),
        if km_score == min(km_score, gmm_score, lac_score):
            print r"{\bf %.6f} &" % km_score,
        else:
            print "%.6f &" % km_score,
        if gmm_score == min(km_score, gmm_score, lac_score):
            print r"{\bf %.6f} &" % gmm_score,
        else:
            print "%.6f &" % gmm_score,
        if lac_score == min(km_score, gmm_score, lac_score):
            print r"{\bf %.6f} \\ %% %d" % (lac_score, lac.k)
        else:
            print r"%.6f \\ %% %d" % (lac_score, lac.k)

        def annotate(minis):
            def _annotate(ax):
                for m in minis:
                    ax.plot([m-1, m, m+1], [.5, .5, .5])
            return _annotate

        continue
        parallel_coordinates_plot(ds.name + ".kmeans.png", x,
                                  means=km.means, stdevs=np.sqrt(km.covars), annotate=annotate(km.minis))
        parallel_coordinates_plot(ds.name + ".lac.png", x,
                                  means=lac.means, stdevs=np.sqrt(lac.covars), annotate=annotate(lac.minis))
        parallel_coordinates_plot(ds.name + ".gmm.png", x,
                                  means=gmm.means, stdevs=np.sqrt(gmm.covars), annotate=annotate(gmm.minis))

    if print_latex:
        print r"\end{tabular}"
    results = np.array(results)


def comparison_plot():
    import pylab as plt

    km = results[1:, 0]
    gmm = results[1:, 1]
    lac = results[1:, 2]

    plt.plot(gmm, lac, 'x')
    plt.plot([0, 80], [0, 80])
    plt.xlabel("gmm")
    plt.ylabel("lac")
    plt.show()


def rank_plot():
    from Orange.evaluation.scoring import graph_ranks
    avgranks = np.mean(np.argsort(np.argsort(results)) + 1, axis=0)
    cd = compute_CD(avgranks, len(results))
    print avgranks, cd
    graph_ranks('ranks.pdf', avgranks, ["kmeans", 'gmm', 'lac'], cd=cd)

print('normalization,reorder,score,ds.name,km_score,gmm_score,lac_score,lac.k')

for normalization in ['01', 'stdev', 'none']:
    for reorder in ['none', 'covariance', 'probability']:
        for score in ['best_triplet_probability']:
            test(chain(continuous_uci_datasets(), GDS_datasets()), print_latex=False,
                 reorder='none', normalization='01', score='best_triplet_probability')
