
from PyQt5.QtWidgets import (
    QLabel, 
    QVBoxLayout, QWidget, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer
from window_functions import centerWindow


class StatisticsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistics Report")
        self.setGeometry(0,0,1200,800)
        self.centerWindow()

            
        # TO DO - vis statistikker osv.
        
       