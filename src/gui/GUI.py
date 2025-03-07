from PyQt5.QtGui import QPixmap  
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from gui.MainWindow import Ui_MainWindow
from controller.Controller import Controller
from shared.Commands import Command
from PIL import ImageQt
import os
class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        self.controller = Controller()
        self.image_path = None
        self.segmented_image = None

        self.action_train_model.triggered.connect(self.on_train_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)


    def on_open_image_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        self.image_path = file_path
        if file_path: 
            pixmap = QPixmap(file_path) 
            self.image_view.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))  

    def on_segment_image_clicked(self):
        if (self.image_path == None):
            raise FileNotFoundError("You must open a file first.")
        self.segmented_image = self.controller.process_command(Command.SEGMENT,self.image_path)
        segmented_image_temp = ImageQt.ImageQt(self.segmented_image)
        pixmap = QPixmap.fromImage(segmented_image_temp)
        self.plot3.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))

    def on_train_model_clicked(self):
        self.controller.process_command(Command.RETRAIN, "data/images", "data/masks")

    def on_load_model_clicked(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        if model_path: 
            self.controller.process_command(Command.LOAD_MODEL, model_path)
        

    def on_export_segmented_clicked(self):
        print("clicked export")
        
        if(self.segmented_image == None):
            raise FileNotFoundError("Error: Found no segmented image")

        file_path, selected_filter = QFileDialog.getSaveFileName(None, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path: 
            # Extract extension from selected filter if not provided
            if not os.path.splitext(file_path)[1]:  # If no extension
                if "PNG" in selected_filter:
                    file_path += ".png"
                elif "JPEG" in selected_filter:
                    file_path += ".jpg"
                else:
                    file_path += ".png"  # Default to PNG
            
            self.segmented_image.save(file_path)
            print(f"Image saved successfully to {file_path}")
        
