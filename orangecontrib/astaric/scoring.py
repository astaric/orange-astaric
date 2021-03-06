from collections import namedtuple
from math import sqrt
from itertools import chain

import datetime
import os
import random
import subprocess
import sys
import numpy as np
import scipy.stats as stats
import sklearn
import sklearn.mixture
import Orange
from Orange.data import Domain, Table, ContinuousVariable, DiscreteVariable
from sklearn.cluster import _k_means
#from sklearn.cluster.k_means_ import _squared_norms

#from orangecontrib.astaric.gmm import em
from Orange.preprocess import Normalize
from orangecontrib.astaric.lac import lac, create_contingencies, MIN_COVARIANCE


def iris():
    yield Table("iris")


def impute(table):
    from Orange.preprocess import SklImpute
    return SklImpute()(table)


def sample(table, p=0.1):
    ind = np.array([1 if random.random() < p else 0 for i in table.X], dtype='bool')
    return Table.from_numpy(table.domain, table.X[ind], table.Y[ind])

def continuous_uci_datasets():
    datasets_dir = os.path.join(os.path.dirname(Orange.__file__), 'datasets')
    for ds in [file for file in os.listdir(datasets_dir) if file.endswith('tab')]:
        if ds in ["adult.tab"]:
            continue
        if ds in ["adult_sample.tab", "horse-colic_learn.tab", "horse-colic_test.tab",
                  "geo-gds360.tab"]:
            continue
        table = open_ds(ds)
        if table is not None:
            table.name = ds
            yield table

def open_ds(ds, filter=True):
    table = Table(ds)
    continuous_features = [a for a in table.domain.attributes if a.is_continuous]
    if not filter or len(continuous_features) > 5:
        print(ds)
        new_table = Table(Domain(continuous_features, [table.domain.class_var]), table)
        impute(new_table)
        new_table.name = ds
        return new_table


def GDS_datasets():
    datasets_dir = '/Users/anze/dev/orange-astaric/orangecontrib/astaric/GDS'
    for ds in [file for file in os.listdir(datasets_dir) if file.startswith('GDS') and file.endswith('tab')]:
        table = Table(os.path.join(datasets_dir, ds))
        continuous_features = [a for a in table.domain.attributes if isinstance(a, ContinuousVariable)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), table)
            impute(new_table)
            new_table.name = ds
            yield new_table


def temporal_datasets():
    datasets_dir = '/Users/anze/dev/orange-astaric/orangecontrib/astaric/temporal'
    for ds in [file for file in os.listdir(datasets_dir) if file.endswith('tab')]:
        table = Table(os.path.join(datasets_dir, ds))
        continuous_features = [a for a in table.domain.attributes if isinstance(a, ContinuousVariable)]
        if len(continuous_features) > 5:
            new_table = Table(Domain(continuous_features), table)
            impute(new_table)
            new_table.name = ds
            yield new_table


Result = namedtuple('results', ['priors', 'means', 'covars', 'k', 'minis'])


def KM(X, k):
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    Y = kmeans.fit_predict(X)

    priors = np.zeros((k,))
    means = kmeans.cluster_centers_
    covars = np.zeros((k, X.shape[1]))
    for j in range(k):
        priors[j] = (Y == j).sum()
        xn = X[Y == j, :] - means[j]
        covars[j] = np.sum(xn ** 2, axis=0) / len(xn)
    priors /= sum(priors)
    return Result(priors, means, covars, k, [])


def LAC(X, k):
    conts = create_contingencies(X, n=10)
    w, means, covars, priors = lac(X.X, conts, k, 100)
    realk = sum(1 for c in covars if (c > MIN_COVARIANCE).all())
    #means, covars = get_cluster_parameters(X.X, get_cluster_weights(priors, means, covars, X.X, crisp=True)[0])

    return Result(priors, np.array(means), np.array(covars), realk, [])

def GMM(X, k):
    gmm = sklearn.mixture.GMM(n_components=k)
    gmm.fit(X)
    return Result(gmm.weights_, gmm.means_, gmm.covars_, k, [])


def LouAUC(ds):
    from Orange.evaluation import LeaveOneOut, AUC
    from Orange.classification import KNNLearner
    results = LeaveOneOut(ds, [KNNLearner()])
    return AUC(results)


def get_cluster_weights(priors, means, covars, x, crisp=True, eps=1e-15):
    k, m = means.shape
    w = np.zeros((len(x), k))
    for j in range(k):
        if any(np.abs(covars[j]) < 1e-15):
            continue
            assert False, 'covars should be fixed'

        det = covars[j].prod()
        inv_covars = 1. / covars[j]
        xn = x - means[j]
        factor = (2.0 * np.pi) ** (x.shape[1]/ 2.0) * det ** 0.5
        w[:, j] = priors[j] * np.exp(np.sum(xn * inv_covars * xn, axis=1) * -.5) / factor
    defined = w.sum(axis=1) > eps
    wsum = w.sum(axis=0)
    wsum[wsum==0] = 1.
    w /= wsum

    if crisp:
        m = w.argmax(axis=1)
        w = np.zeros(w.shape)
        w[np.arange(len(w)), m] = 1.

    return w, defined

def get_cluster_parameters(x, w):
    wsums = w.sum(axis=0)[:, None]
    wsums[wsums == 0] = 1.
    # weighted sum of rows, divided by sum of weights (for each cluster)
    means = (w.T[:, :, None] * x[None, :, :]).sum(axis=1) / wsums
    covars = (w.T[:, :, None] * (x[None, :, :] - means[:, None, :]) ** 2).sum(axis=1) / wsums

    return means, covars





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
            x = list(range(len(mean))) + list(range(len(mean)))[::-1]

            poly = Polygon(list(zip(x, y)), facecolor=colors[i], edgecolor='none', alpha=.5)
            plt.gca().add_patch(poly)

            y = [mean[j] + stdev[j] / 2 for j in range(len(mean))]
            y += [mean[j] - stdev[j] / 2 for j in range(len(mean))][::-1]
            x = list(range(len(mean))) + list(range(len(mean)))[::-1]

            poly = Polygon(list(zip(x, y)), facecolor=colors[i], edgecolor='none', alpha=.8)
            plt.gca().add_patch(poly)

    ax.set_xlim([-.01, X.shape[1] - 0.93])
    annotate(ax)
    ax.axis('off')
    plt.savefig(os.path.join('output', filename), bbox_inches='tight')
    plt.close()


def covariance_score(result, x):
    means = x.mean(axis=0)
    xn = x - means
    stdev = np.sqrt(np.sum(xn ** 2, axis=0) / len(xn))

    return np.sqrt(result.covars / stdev).sum() / xn.shape[1]

def corrected_covariance_score(result, x):
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

def silhouette_score(results, x):
    k = results.means.shape[0]
    priors = np.ones((k,)) / k
    w = get_cluster_weights(priors, results.means, results.covars, x)
    labels = w.argmax(axis=1)
    return sklearn.metrics.silhouette_score(x, labels)

def silhouette_d_score(results, x, defined_=None):
    w, defined = get_cluster_weights(results.priors, results.means, results.covars, x)
    if defined_ is not None:
        defined = defined_
    labels = w.argmax(axis=1)
    x = x[defined, :]
    labels = labels[defined]

    class R(float): pass
    if len(np.unique(labels)) < 2:
        r = R(-1.)
        r.defined = defined
        return r

    result = R(sklearn.metrics.silhouette_score(x, labels))
    result.defined = defined
    return result

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


def reorder_attributes_covariance(ds, k=0):
    cov = np.cov(ds.X, rowvar=0)
    #cov = np.abs(cov)
    order = tuple(reorder_features_by_similarity(cov))
    return ds[:, order]


def reorder_attributes_probability_score(ds, k):
    m = ds.X.shape[1]
    gmm = GMM(ds.X, k)
    probability = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i >= j:
                continue
            probability[i, j] = probability[j, i] = \
                score_dimension_pair(ds.X, gmm, (i, j))
    order = tuple(reorder_features_by_similarity(probability))
    return ds[:, order]



def test(datasets=(),
         normalization="stdev", reorder='none', score='probability',
         print_latex=True, k=10, eps=1e-15):
    global results
    #print("%% normalization=%s,reorder=%s,score=%s, %%" % (normalization, reorder, score))

    if print_latex:
        print(r"\begin{tabular}{ l r r r }")
        print(r"dataset & S(k-means) & S(gmm)& S(lac) \\")
        print(r"\hline")
    results = []
    #for ds in [Table('vehicle', name='vehicle')]:
    #for ds in GDS_datasets():
    for ds2 in datasets:
        np.random.seed(42)
    #for ds in temporal_datasets():
        ds = impute(ds2)
        ds.name = ds2.name
        n_steps = 99

        if normalization == 'none' or normalization is None:
            pass
        elif normalization == '01':
            ds = Normalize(norm_type=Normalize.NormalizeBySpan)(ds)
        elif normalization == 'stdev':
            ds = Normalize(norm_type=Normalize.NormalizeBySD)(ds)
        else:
            raise AttributeError('Unknonwn normalization type "%s"' % (normalization,))

        if reorder == 'none' or reorder is None:
            pass
        elif reorder == 'shuffle':
            np.random.shuffle(x.T)
        elif reorder == 'covariance':
            ds = reorder_attributes_covariance(ds)
        elif reorder == 'probability':
            ds = reorder_attributes_probability_score(ds, k)
        else:
            raise AttributeError('Unknown feature reordering type "%s"' % (reorder,))

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
        elif score == 'silhouette':
            scorer = silhouette_score
        elif score == 'silhouette_d':
            scorer = silhouette_d_score
        else:
            raise AttributeError('Unknown scorer type "%s"' % (score,))


        all_lac_scores = []
        for n in range(10,11):
            #print('.', end='')
            lac = LAC(ds, k)
            all_lac_scores.append((lac.k, scorer(lac, ds.X)))

#        print()
#        for i in range(10, 11):
#            lac_scores = [s for k, s in all_lac_scores if k == i]
#            print("lac, %i, %f, %f, %f, %f" % (i, min(lac_scores or [0]), min([l for l in lac_scores if l] or [0]), max(lac_scores or [0]), sum([l for l in lac_scores if l] or [0]) / len([l for l in lac_scores if l] or [0])))

        realk = max(k for k, s in all_lac_scores)
        #if lac.k < 2:
        #    continue

        try:
            lac = LAC(ds, realk)
            km = KM(ds.X, realk)
            gmm = GMM(ds.X, realk)
        except:
            print("Error")
            continue

        lac_score = scorer(lac, ds.X)
        opts = dict(defined_=lac_score.defined) if hasattr(lac_score, 'defined') else {}
        km_score = scorer(km, ds.X)
        km_score_d = scorer(km, ds.X, **opts)
        gmm_score = scorer(gmm, ds.X)
        gmm_score_d = scorer(gmm, ds.X, **opts)

        knn_score = LouAUC(ds)

        results.append((km_score, gmm_score, lac_score))
        if not print_latex:
            print("dataset: %s (%s rows, %s features)" % (ds.name, len(ds), len(ds.domain)))
            print("normalization: ", normalization)
            print("reorder: ", reorder)
            print("scoring function: ", score)
            print("----------------")
            print("k-means:           %.5f" % km_score)
            print("gmm:               %.5f" % gmm_score)
            print("k-means (dropout): %.5f" % km_score_d)
            print("gmm (dropout):     %.5f" % gmm_score_d)
            print("lac:               %.5f" % lac_score)
            print("----------------")
            print("knn AUC (lou)      %.5f" % knn_score)
            print("----------------")
            print("k=%s, dropout %s (%.1f%%)" % (realk, sum(~lac_score.defined), (sum(~lac_score.defined) / len(
                ds.X)) *
                  100))
            print()
            print()

            # print("%s,%s,%s,%s,%f,%f,%f,%i,%f,%f" % (normalization, reorder, score, ds.name, km_score, gmm_score,
            #                                       lac_score, realk, sum(~lac_score.defined), sum(~lac_score.defined) /
            #                                       len(ds.X)))

        w1, _ = get_cluster_weights(lac.priors, lac.means, lac.covars, ds.X, crisp=False)
        w2, _ = get_cluster_weights(lac.priors, lac.means, lac.covars, ds.X, crisp=True)
        w2 = np.argmax(w2, axis=1)[:, None]

        domain = Domain([ContinuousVariable("p%d" % i) for i in range(w1.shape[1])])
        probs = Table(domain, w1)
        labels = Table(Domain([DiscreteVariable("label", values=list(range(k)))]), w2)

        ds2.name = ds2.name.replace("/", "_")
        ds.name = ds2.name.replace("/", "_")
        tbl = Table.concatenate((ds, probs, labels))
        tbl.save(os.path.join('output', ds2.name + ".lac.tab"))




        def annotate(minis):
            def _annotate(ax):
                for m in minis:
                    ax.plot([m-1, m, m+1], [.5, .5, .5])
            return _annotate

        parallel_coordinates_plot(ds.name + ".kmeans.pdf", ds.X,
                                  means=km.means, stdevs=np.sqrt(km.covars), annotate=annotate(km.minis))
        parallel_coordinates_plot(ds.name + ".lac.pdf", ds.X,
                                  means=lac.means, stdevs=np.sqrt(lac.covars), annotate=annotate(lac.minis))
        parallel_coordinates_plot(ds.name + ".gmm.pdf", ds.X,
                                  means=gmm.means, stdevs=np.sqrt(gmm.covars), annotate=annotate(gmm.minis))

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab
        import math

        #iris = Table("wine")
        #for m in range(lac.means.shape[1]):
        #    plt.clf()
        #    for p, mean, variance, c in zip(lac.priors, lac.means[:, m], lac.covars[:, m], "grb"):
        #        sigma = math.sqrt(variance)
        #        x = np.linspace(0,1,100)
        #         plt.plot(x,p * mlab.normpdf(x, mean,sigma), color=c)
        #     plt.plot(ds.X[:, m].ravel() + np.random.random(len(ds)) * 0.02, [0.05]*len(ds.X) + iris.Y.ravel() * 0.1,
        #              "k|")
        #     plt.ylabel("pdf")
        #     plt.xlabel(iris.domain[m].name)
        #     plt.savefig("axis-%d.pdf" % m)

    if print_latex:
        print(r"\end{tabular}")
    results = np.array(results)
    #comparison_plot()


def comparison_plot():
    import pylab as plt

    km = results[1:, 0]
    gmm = results[1:, 1]
    lac = results[1:, 2]

    plt.plot(gmm, lac, 'x')
    plt.plot([-2, 2], [-2, 2])
    plt.xlabel("gmm")
    plt.ylabel("lac")
    plt.show()


def rank_plot():
    from Orange.evaluation.scoring import graph_ranks
    avgranks = np.mean(np.argsort(np.argsort(results)) + 1, axis=0)
    cd = compute_CD(avgranks, len(results))
    print(avgranks, cd)
    graph_ranks('ranks.pdf', avgranks, ["kmeans", 'gmm', 'lac'], cd=cd)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("Evaluate LAC")
    parser.add_argument("datasets", metavar="dataset", nargs="+", help="dataset to run the tests on")
    parser.add_argument("-r", dest="reorder", default="none",
                        help="reorder features (none, shuffle, covariance, probability)")
    parser.add_argument("-n", dest="normalization", default="01",
                        help="normalize features (none, 01, stdev)")
    parser.add_argument("-s", dest="score", default="silhouette_d",
                        help="scoring function (covariance, probability, global_probability, best_triplet, "
                             "best_triplet_probability, silhouette, silhouette_d)")
    parser.add_argument("-k", dest="k", type=int, default=10,
                        help="number of clusters")
    parser.add_argument("-e", dest="e", type=float, default=1e-15,
                        help="dropout eps")
    args = parser.parse_args()

    def datasets_generator():
          for d in args.datasets:
            table = open_ds(d, filter=False)
            table.name = d
            yield table
    datasets = datasets_generator()

    if os.path.exists('output'):
        timestamp = datetime.datetime.fromtimestamp(os.path.getctime('output'))
        os.rename('output', 'output - %s' % timestamp)
    os.mkdir('output')

    # Log std output to file
    tee = subprocess.Popen(["tee", "output/output.txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    test(datasets, print_latex=False,
         reorder=args.reorder, normalization=args.normalization, score=args.score, k=args.k, eps=args.e)
