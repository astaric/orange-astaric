from Orange.data import Table
from Orange.OrangeCanvas.registry.description import Default
from Orange.OrangeWidgets.OWBaseWidget import AttributeList
from Orange.OrangeWidgets.OWColorPalette import ColorPaletteGenerator
from Orange.OrangeWidgets.OWGUI import label, spin, widgetBox, checkBox

from Orange.OrangeWidgets.Visualize.OWParallelCoordinates import OWParallelCoordinates as orangeOWParallelCoordinates
from PyQt4.QtGui import QApplication
from orangecontrib.astaric.widgets.owparallelgraph import OWParallelGraph

NAME = "Extended parallel coordinates Data"
DESCRIPTION = "Parallel coordinates plot with optional grouping."
LONG_DESCRIPTION = ""
ICON = "icons/ParallelCoordinates.svg"
PRIORITY = 1150
AUTHOR = "Anze Staric"
AUTHOR_EMAIL = "anze.staric@fri.uni-lj.si"
INPUTS = [
    ("Data", Table, "setData", Default),
    ("Features", AttributeList, 'setShownAttributes')
]
OUTPUTS = [
    ("Selected Data", Table),
    ("Other Data", Table),
    ("Features", AttributeList)
]


class OWParallelCoordinates(orangeOWParallelCoordinates):
    settingsList = orangeOWParallelCoordinates.settingsList + \
                   ['graph.group_lines', 'graph.number_of_groups', 'graph.number_of_steps']

    def __init__(self, parent=None, signalManager=None):
        super(OWParallelCoordinates, self).__init__(parent, signalManager)

        self.mainArea.layout().removeWidget(self.graph)
        self.graph = OWParallelGraph(self, self.mainArea)
        self.mainArea.layout().addWidget(self.graph)

        self.graph.jitterSize = 10
        self.graph.showDistributions = 1
        self.graph.showStatistics = 0
        self.graph.showAttrValues = 1
        self.graph.useSplines = 0
        self.graph.enabledLegend = 1

        self.loadSettings()

        dlg = self.createColorDialog()
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        self.graph.setCanvasBackground(dlg.getColor("Canvas"))
        apply([self.zoomSelectToolbar.actionZooming, self.zoomSelectToolbar.actionRectangleSelection,
               self.zoomSelectToolbar.actionPolygonSelection][self.toolbarSelection], [])
        self.cbShowAllAttributes()

        self.add_group_settings(self.SettingsTab)

    def add_group_settings(self, parent):
        box = widgetBox(parent, "Groups", orientation="vertical")
        box2 = widgetBox(box, orientation="horizontal")
        checkBox(box2, self, "graph.group_lines", "Group lines into", tooltip="Show clusters instead of lines",
                 callback=self.updateGraph)
        spin(box2, self, "graph.number_of_groups", 0, 30, callback=self.updateGraph)
        label(box2, self, "groups")
        box2 = widgetBox(box, orientation="horizontal")
        spin(box2, self, "graph.number_of_steps", 0, 100, label="In no more than", callback=self.updateGraph)
        label(box2, self, "steps")

#test widget appearance
if __name__ == "__main__":
    import sys
    from Orange.data import Table

    a = QApplication(sys.argv)
    ow = OWParallelCoordinates()
    ow.show()
    ow.graph.discPalette = ColorPaletteGenerator(rgbColors=[(127, 201, 127), (190, 174, 212), (253, 192, 134)])
    data = Table("/Users/anze/dev/orange3/Orange/widgets/visualize/edt-all-vs-zero.tab")
    ow.setData(data)
    ow.handleNewSignals()

    a.exec_()

    ow.saveSettings()