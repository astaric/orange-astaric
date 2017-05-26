from csv import reader, writer

import numpy as np
from scipy.stats import rankdata

from Orange.evaluation.scoring import compute_CD, graph_ranks


with open("output/results.csv") as f:
    csv = iter(reader(f))
    header = next(csv)
    lines = list(csv)
    lines.sort(key=lambda x:x[1], reverse=True)

    header.extend(["{} Rank".format(m) for m in header[1:6]])
    for line in lines:
        ranks = rankdata([-float(x) for x in line[1:6]], method="min")
        line.extend(ranks)

with open("output/sorted_results.csv", "w") as f:
    csv = writer(f)
    csv.writerow(header)
    for line in lines:
        csv.writerow(line)


def draw_rank_comparison(lines, name=""):
    ranks = np.array([x[-5:] for x in lines])
    avg_ranks = ranks.mean(axis=0)
    cd = compute_CD(avg_ranks, len(lines))

    graph_ranks(avg_ranks, header[1:6], cd=cd,
                filename="output/ranks-{}.png".format(name))


draw_rank_comparison(lines, "all")

draw_rank_comparison(lines[:30], "best")
draw_rank_comparison(lines[30:60], "average")
draw_rank_comparison(lines[60:], "worst")
