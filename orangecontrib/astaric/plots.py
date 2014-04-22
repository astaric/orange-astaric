from matplotlib.patches import Polygon, Ellipse
import numpy as np
from orangecontrib.astaric.gmm import em
import pylab as plt

from Orange.data import Table

colors = [
    '#7fc97f',
    '#beaed4',
    '#fdc086',
    '#4f79b3',
]


def parallel_coordinates_plot(filename, X, means=None, stdevs=None, annotate=lambda ax: None):
    fig, ax = plt.subplots()
    for y in X:
        x = np.array(range(len(y)))
        ax.plot(x, y, color='k', alpha=.1)

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


def plot_showcase():
    ds = Table('showcase')
    X, = ds.to_numpy("a")
    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)

    def annotate_critical_areas(ax):
        ax.add_artist(Ellipse(xy=(1., 0.6,), width=.1, height=0.3, facecolor='none'))
        ax.annotate('1.', xy=(1, 0.6), xytext=(0.5, 0.4),
                    arrowprops=dict(facecolor='white', shrink=0.05))
        ax.add_artist(Ellipse(xy=(2., 0.725,), width=.1, height=0.5, facecolor='none'))
        ax.add_artist(Ellipse(xy=(3., 0.8,), width=.1, height=0.4, facecolor='none'))
        ax.annotate('2.', xy=(2., 0.725), xytext=(2.5, 0.5),
                    arrowprops=dict(facecolor='white', shrink=0.05))
        ax.annotate('2.', xy=(3., 0.725), xytext=(2.5, 0.5),
                    arrowprops=dict(facecolor='white', shrink=0.05))

    parallel_coordinates_plot("showcase-lines.pdf", X, annotate=annotate_critical_areas)


def plot_showcase_with_clusters():
    ds = Table('showcase')
    X, = ds.to_numpy("a")
    w, means, covars, priors = em(X, 3, 100)
    stdevs = np.sqrt(covars)

    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)
    means = (means - m) / (M - m)
    stdevs /= M - m

    parallel_coordinates_plot("showcase-clusters.pdf", X, means, stdevs)


def plot_wine_with_lac():
    ds = Table('wine')
    X, = ds.to_numpy("a")
    X = X[:, (3,4,5,6,9)]
    w, means, covars, priors = em(X, 3, 100)
    stdevs = np.sqrt(covars)

    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)
    means = (means - m) / (M - m)
    stdevs /= M - m

    parallel_coordinates_plot("wine-lac.pdf", X, means, stdevs)


def plot_wine_with_km():
    ds = Table('wine')
    X, = ds.to_numpy("a")
    X = X[:, (3,4,5,6,9)]
    w, means, covars, priors = em(X, 3, 0)
    stdevs = np.sqrt(covars)

    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)
    means = (means - m) / (M - m)
    stdevs /= M - m

    parallel_coordinates_plot("wine-km.pdf", X, means, stdevs)


def plot_yeast_with_lac():
    ds = Table('yeast-class-RPR')
    from Orange.feature.imputation import AverageConstructor
    ds = AverageConstructor()(ds)(ds)

    X, = ds.to_numpy("a")
    X = X[:, (72, 73, 74, 75, 76, 77, 78)]
    w, means, covars, priors = em(X, 4, 100)
    stdevs = np.sqrt(covars)

    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)
    means = (means - m) / (M - m)
    stdevs /= M - m

    parallel_coordinates_plot("yeast-lac.pdf", X, means, stdevs)


def plot_yeast_with_km():
    ds = Table('yeast-class-RPR')
    from Orange.feature.imputation import AverageConstructor
    ds = AverageConstructor()(ds)(ds)

    X, = ds.to_numpy("a")
    X = X[:, (72, 73, 74, 75, 76, 77, 78)]
    w, means, covars, priors = em(X, 4, 0)
    stdevs = np.sqrt(covars)

    m, M = X.min(axis=0), X.max(axis=0)
    X = (X - m) / (M - m)
    means = (means - m) / (M - m)
    stdevs /= M - m

    parallel_coordinates_plot("yeast-km.pdf", X, means, stdevs)


plot_showcase()




