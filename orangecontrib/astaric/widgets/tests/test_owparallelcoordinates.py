import unittest
import sys

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest

from Orange.data import Table
from Orange.feature import Discrete, Continuous
from Orange.orange import Domain


from orangecontrib.astaric.widgets.OWParallelCoordinates import OWParallelCoordinates


class OWParallelCorodinatesTest(unittest.TestCase):
    app = None

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    def setUp(self):
        self.widget = OWParallelCoordinates()
        self.reset_widget()

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def connect_data(self):
        attrs = [Discrete("D%s" % i, values="01234") for i in range(3)] + \
                [Continuous("C%s" % i) for i in range(3)] + \
                [Discrete("Class", values="01")]
        domain = Domain(attrs)
        data = [
            [0, 2, 2, 3., 7., 1., 0],
            [1, 3, 2, 2., 6., 1., 0],
            [2, 1, 2, 5., 8., 1., 1],
        ]

        self.table = Table(domain, data)
        self.widget.setData(self.table)
        self.widget.handleNewSignals()

    def select_item(self, list_box, index, modifiers=Qt.NoModifier):
        rect = list_box.visualItemRect(list_box.item(index))
        QTest.mouseClick(list_box.viewport(), Qt.LeftButton, modifiers, rect.center())

    def select_items(self, list_box, indices):
        if indices:
            self.select_item(list_box, indices[0])
        for index in indices[1:]:
            self.select_item(list_box, index, Qt.ShiftModifier)

    def click(self, control):
        QTest.mouseClick(control, Qt.LeftButton)

    def debug_show_widget(self):
        self.widget.show()
        self.app.exec_()

    def reset_widget(self):
        self.widget.setData(None)
        self.widget.handleNewSignals()

        self.widget.showAllAttributes = False
        self.widget.graph.showStatistics = False
        self.widget.middleLabels = "No labels"
        self.widget.graph.showDistributions = False
