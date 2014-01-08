#
# OWParallelGraph.py
#
from math import sqrt
from Orange.OrangeWidgets.OWGraphTools import PolygonCurve
from Orange.OrangeWidgets.Visualize.OWParallelGraph import OWParallelGraph as orangeOWParallelGraph
from Orange.data.preprocess.scaling import get_variable_values_sorted
from Orange.feature import Continuous
from PyQt4.QtGui import QPen, QBrush, QColor
from orangecontrib.astaric.gmm import em


class OWParallelGraph(orangeOWParallelGraph):
    group_lines = False
    number_of_groups = 10
    number_of_steps = 30

    def __init__(self, parallelDlg, parent=None, name=None):
        super(OWParallelGraph, self).__init__(parallelDlg, parent, name)
        self.groups = {}

    def updateData(self, attributes, midLabels=None, updateAxisScale=1):
        super(OWParallelGraph, self).updateData(attributes, midLabels, updateAxisScale)

        if self.group_lines:
            self.draw_groups()

        self.replot(True)

    def compute_groups(self):
        key = (tuple(self.visualizedAttributes), self.number_of_groups, self.number_of_steps)
        if key not in self.groups:
            indices = [self.attribute_name_index[label] for label in self.visualizedAttributes]
            X = self.original_data[indices].T
            w, mu, sigma, phi = em(X, self.number_of_groups, self.number_of_steps)
            self.groups[key] = phi, mu, sigma
        return self.groups[key]

    def draw_groups(self):
        if self.original_data is None:
            return

        phis, mus, sigmas = self.compute_groups()
        indices = [self.attribute_name_index[label] for label in self.visualizedAttributes]

        diff, mins = [], []
        for i in indices:
            if isinstance(self.data_domain[i], Continuous):
                diff.append(self.domain_data_stat[i].max - self.domain_data_stat[i].min or 1)
                mins.append(self.domain_data_stat[i].min)
            else:
                attribute_values = get_variable_values_sorted(self.data_domain[i])
                attr_len = len(attribute_values)
                diff.append(attr_len)
                mins.append(1/(2 * attr_len))

        for j, (phi, cluster_mus, cluster_sigma) in enumerate(zip(phis, mus, sigmas)):
            for i, (mu1, sigma1, mu2, sigma2), in enumerate(
                    zip(cluster_mus, cluster_sigma, cluster_mus[1:], cluster_sigma[1:])):
                nmu1 = (mu1 - mins[i]) / diff[i]
                nmu2 = (mu2 - mins[i + 1]) / diff[i + 1]
                nsigma1 = sqrt(sigma1) / diff[i]
                nsigma2 = sqrt(sigma2) / diff[i + 1]
                color = self.discPalette.getRGB(j)

                self.draw_group(i, nmu1, nmu2, nsigma1, nsigma2, phi, color)
        self.replot()

    def draw_group(self, i, nmu1, nmu2, nsigma1, nsigma2, phi, color):
        xs2s = [i, i, i + 1, i + 1, i]
        ys2s = [nmu1 - nsigma1, nmu1 + nsigma1, nmu2 + nsigma2, nmu2 - nsigma2, nmu1 - nsigma1]
        xs1s = xs2s
        ys1s = [nmu1 - .5 * nsigma1, nmu1 + .5 * nsigma1, nmu2 + .5 * nsigma2, nmu2 - .5 * nsigma2, nmu1 - .5 * nsigma1]

        if isinstance(color, tuple):
            color = QColor(*color)
        color.setAlphaF(.3)

        outercurve = PolygonCurve(QPen(QColor(0, 0, 0, 0)), QBrush(color), xData=xs2s, yData=ys2s)
        outercurve.attach(self)
        innercurve = PolygonCurve(QPen(color), QBrush(color), xData=xs1s, yData=ys1s)
        innercurve.attach(self)

    def replot(self, really=False):
        if not really:
            return

        super(OWParallelGraph, self).replot()