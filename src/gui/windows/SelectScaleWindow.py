from PyQt5 import QtCore, QtGui, QtWidgets
from gui.ui.SelectScaleUI import Ui_SelectScale
from PyQt5.QtWidgets import QGraphicsScene

class SelectScaleWindow(QtWidgets.QWidget, Ui_SelectScale):
    scale_bar_set_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.image_scene = QGraphicsScene(self)
        self.image.setScene(self.image_scene)
        self.coordinates = []

    def mousePressEvent(self, event):
        mouse_pos = event.pos()
        x = mouse_pos.x()

        self.coordinates.append(x)
        if len(self.coordinates) == 2:
            self.scale_bar_set_signal.emit(self.coordinates)
            self.close()
    