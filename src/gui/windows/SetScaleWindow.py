from PyQt5.QtWidgets import QDialog  

from src.gui.ui.SetScaleUI import Ui_set_scale_window


class SetScaleWindow(QDialog, Ui_set_scale_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
