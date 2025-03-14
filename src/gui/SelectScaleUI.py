from PyQt5 import QtCore, QtGui, QtWidgets
from gui.SelectScaleWindow import Ui_SelectScaleWindow
from PyQt5.QtWidgets import QGraphicsScene

class SelectScaleUI(QtWidgets.QWidget, Ui_SelectScaleWindow):
    scale_bar_set_signal = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.image_scene = QGraphicsScene(self)
        self.image.setScene(self.image_scene)
        self.coordinates = []

    def mousePressEvent(self, event):
        # Get the position of the mouse click
        mouse_pos = event.pos()
        x, y = mouse_pos.x(), mouse_pos.y()
        
        # Store the coordinates
        self.coordinates.append(x)
        if len(self.coordinates) == 2:
            self.scale_bar_set_signal.emit(self.coordinates)
            self.close()
    