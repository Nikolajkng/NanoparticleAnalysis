
from PyQt5.QtWidgets import (
    QLabel, 
    QVBoxLayout, QWidget, QDesktopWidget
)
from PyQt5.QtCore import Qt


class AnalysisWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysing Window")
        self.setGeometry(400, 200, 400, 300)
        self.centerWindow()

        layout = QVBoxLayout()
        label = QLabel("Analysis in progress...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
        
        # TO DO - vis statistikker osv.
        
       

    def centerWindow(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft()) 