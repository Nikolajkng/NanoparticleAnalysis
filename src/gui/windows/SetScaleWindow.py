from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog  
from shared.ModelConfig import ModelConfig
from gui.ui.SetScaleUI import Ui_set_scale_window


class SetScaleWindow(QDialog, Ui_set_scale_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
