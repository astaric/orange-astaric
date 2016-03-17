import os
import pickle
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.evaluation import TestOnTrainingData, CA
from Orange.classification.knn import KNNLearner
from Orange.widgets.visualize.owscattermap import compute_chi_squares


random.seed(42)
n_of_bins = 2**4
n_of_classes = 2



def generate_two_clusters(field, x, y):
    area = field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins, :]

    area[2:5, 2:5, :] = [5, 0]
    area[10:13, 10:13, :] = [0, 5]
    #area[0, 0, :] = [5, 0]
    #area[1, 1, :] = [0, 5]


def generate_random_dataset(k=10, n_of_classes=2):
    random.seed(42)
    field = np.zeros((n_of_bins ** 2, n_of_bins ** 2, n_of_classes))

    for i in range(k):
        x=random.randint(0, n_of_bins-1)
        y=random.randint(0, n_of_bins-1)
        print(y, x)

        generate_two_clusters(field, x, y)

    return field



def generate_k_regions(k=10, n=40, covariance=150, w=1):
    random.seed(42)

    field = np.zeros((n_of_bins ** 2, n_of_bins ** 2, k))

    for c in range(k):
        mx = n_of_bins**2
        mu=np.array([random.randint(int(mx/4), int(3/4*mx)),
                     random.randint(int(mx/4), int(3/4*mx))])

        data = np.random.multivariate_normal(mu, np.diag(np.array([covariance, covariance])), size=n)
        for y, x in data:
            field[int(y), int(x), c] += 1

    return field



def unsharpen(field, sharpened=()):
    new_field = np.zeros(field.shape)

    for y in range(n_of_bins):
        for x in range(n_of_bins):
            if (y, x) not in sharpened:
                new_field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins, :] = \
                    field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins].mean(axis=0).mean(axis=0)
            else:
                new_field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins, :] = \
                    field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins, :]

    return new_field

def create_examples(x, y, counts):
    for c, v in enumerate(counts):
        for _ in range(int(v)):
            yield (x, y, c)


def materialize(field, sharpened=()):
    examples = []

    for y1 in range(n_of_bins):
        for x1 in range(n_of_bins):
            if (y1, x1) not in sharpened:
                counts = field[y1*n_of_bins:(y1+1)*n_of_bins, x1*n_of_bins:(x1+1)*n_of_bins].sum(axis=0).sum(axis=0)
                examples.extend(create_examples((x1+.5)*n_of_bins, (y1 * .5)*n_of_bins, counts))
            else:
                for y2 in range(n_of_bins):
                    for x2 in range(n_of_bins):
                        counts = field[y1*n_of_bins + y2, x1*n_of_bins + x2]
                        examples.extend(create_examples(x1*n_of_bins + x2 +.5, y1*n_of_bins + y2 +.5, counts))

    return Table(Domain([ContinuousVariable("x"), ContinuousVariable("y")],
                        [DiscreteVariable("class", values=["1", "2"])]),
                 np.array(examples))


fig1 = plt.figure()
def plot_field(field, a, b, c):
    h, w, _ = field.shape
    ax1 = fig1.add_subplot(a,b,c, aspect='equal')

    for i in range(h):
        for j in range(w):
            counts = field[i, j]

            if not counts.any():
                continue
            else:
                color = 'brggitcmyk'[counts.argmax()]

            ax1.add_patch(
                patches.Rectangle(
                    (j/w, i/h),   # (x,y)
                    1/w,          # width
                    1/h,          # height
                    edgecolor=None,
                    color=color
                )
            )




def evaluate(field, method):
    ordering = method(field)

    results = []

    for i in range(len(ordering)):
        a = materialize(field, ordering[:i+1])#, sharpened=)

        res = TestOnTrainingData(a, [KNNLearner()])
        results.append(CA(res))
        print(results[-1])
    return np.array(results).flatten()


def compute_chi_squares(observes):
    """Compute chi2 scores of given observations.

    Assumes that data is generated by two independent distributions,
    one for rows and one for columns and estimate distribution parameters
    from data.

    Parameters
    ----------
    observes : numpy array with dimensions (N_CLASSES * N_ROWS * N_COLUMNS)
        Multiple contingencies containing observations for multiple classes.
    """
    CLASSES, COLS, ROWS = 0, 1, 2

    n = observes.sum((ROWS, COLS), keepdims=True)
    row_sums = observes.sum(ROWS, keepdims=True)
    col_sums = observes.sum(COLS, keepdims=True)
    estimates = row_sums * col_sums / n

    chi2 = np.nan_to_num(np.nansum((observes - estimates)**2 / estimates, axis=CLASSES))
    return chi2


def random_selection(field):
    fields = [(i, j) for i in range(n_of_bins) for j in range(n_of_bins)]
    random.shuffle(fields)
    return fields

def linear_selection(field):
    return [(i, j) for i in range(n_of_bins) for j in range(n_of_bins)]

def my_selection(field):
    def get_dist():
        dist = np.zeros((field.shape[2], n_of_bins, n_of_bins))
        for c in range(field.shape[2]):
            for y in range(n_of_bins):
                for x in range(n_of_bins):
                    dist[c][y][x] = field[y*n_of_bins:(y+1)*n_of_bins, x*n_of_bins:(x+1)*n_of_bins, c].sum()
        return dist

    dist = get_dist()
    chi2 = compute_chi_squares(dist)

    fields = [(i, j) for i in range(n_of_bins) for j in range(n_of_bins)]
    fields.sort(key=lambda x: chi2[x[0]][x[1]], reverse=True)

    return fields


field = generate_random_dataset(10, n_of_classes=2)
#field = generate_k_regions(k=2, n=50, covariance=500, w=2)
#field = generate_random_dataset(10, n_of_classes=2)


plot_field(unsharpen(field), 2, 2, 1)
plot_field(field, 2, 2, 2)
fig1.savefig('sm.pdf', bbox_inches='tight')

if os.path.exists('results.pck'):
    with open('results.pck', 'rb') as f:
        r = pickle.load(f)
        l = pickle.load(f)
        m = pickle.load(f)
else:
    print('random')
    r = evaluate(field, random_selection)
    print('linear')
    l = evaluate(field, linear_selection)
    print('scattermap')
    m = evaluate(field, my_selection)

    with open('results.pck', 'wb') as f:
        pickle.dump(r, f)
        pickle.dump(l, f)
        pickle.dump(m, f)



fig1 = plt.figure()
plt.plot(r, label='random')
plt.plot(l, label='linear')
plt.plot(m, label='scattermap')
plt.legend(loc='lower right')
plt.ylabel('CA')
plt.xlabel('iterations')
fig1.savefig('results.pdf', bbox_inches='tight')

print('done')





