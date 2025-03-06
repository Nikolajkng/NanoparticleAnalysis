from gui.MainWindow import Ui_MainWindow
import sys
import os

from controller.Controller import Controller

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap  # QPixmap belongs to QtGui
from PyQt5.QtWidgets import QFileDialog, QMainWindow  # QFileDialog belongs to QtWidgets
from shared.Commands import Command
class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        self.controller = Controller()
        self.image_path = None

        self.action_train_model.triggered.connect(self.onTrainModelClicked)
        self.action_open_image.triggered.connect(self.onOpenImageClicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.onSegmentImageClicked)

    def onOpenImageClicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        self.image_path = file_path
        if file_path: 
            pixmap = QPixmap(file_path) 
            self.image_view.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))  

    def onSegmentImageClicked(self):
        if (self.image_path == None):
            raise FileNotFoundError("You must open a file first.")
        
        segmented_image_file_path = self.controller.segment_image(file_path=self.image_path)

        if (segmented_image_file_path):
            pixmap = QPixmap(segmented_image_file_path)
            self.plot3.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))

    def onTrainModelClicked(self):
        self.controller.process_command(Command.RETRAIN, "data/images", "data/masks")



