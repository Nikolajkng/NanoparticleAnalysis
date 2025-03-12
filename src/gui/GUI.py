import csv
import threading
from PyQt5.QtGui import QPixmap  
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from gui.MainWindow import Ui_MainWindow
from controller.Controller import Controller
from shared.Commands import Command
from PIL import ImageQt
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QFileDialog, QMessageBox, QApplication
from functools import partial 

class GUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        self.controller = Controller()
        self.image_path = None
        self.segmented_image = None
        self.csv_file = None


        self.action_train_model.triggered.connect(self.on_train_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)
        self.actionExport_Data_as_csv.triggered.connect(self.on_export_data_csv_clicked)


    def on_open_image_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        self.image_path = file_path
        if file_path: 
            pixmap = QPixmap(file_path) 
            self.image_view.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))  

    def on_segment_image_clicked(self):
        if (self.image_path == None):
            self.messageBox("Segmentation failed: No image found")
        self.segmented_image = self.controller.process_command(Command.SEGMENT,self.image_path)
        segmented_image_temp = ImageQt.ImageQt(self.segmented_image)
        pixmap = QPixmap.fromImage(segmented_image_temp)
        self.plot3.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))


    def on_train_model_clicked(self):
        try:
            self.controller.process_command(Command.RETRAIN, "data/images", "data/masks")
        
        # TODO: Separat thread fucker live-plot op for hold-out op.
        # try:
        #     train_thread = threading.Thread(
        #         target=partial(self.controller.process_command, Command.RETRAIN, "data/images", "data/masks"),
        #         daemon=True)
        #     train_thread.start()
            
            self.messageBoxTraining("success")
        except:
            self.messageBoxTraining("")


    def on_load_model_clicked(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        if model_path: 
            self.controller.process_command(Command.LOAD_MODEL, model_path)
        

    def on_export_segmented_clicked(self):
        if(self.segmented_image == None):
            self.messageBox("Export failed: No segmented image was found to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(None, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path: 
            if not os.path.splitext(file_path)[1]: 
                if "PNG" in selected_filter:
                    file_path += ".png"
                elif "JPEG" in selected_filter:
                    file_path += ".jpg"
                else:
                    file_path += ".png" 
            
            self.segmented_image.save(file_path)            
            self.messageBox("success")
        else:
            self.messageBox("Error: File path is not selected.")
            
          
    def on_export_data_csv_clicked(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save CSV")
        file_dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        file_dialog.setOptions(options)
        file_path, selected_filter = file_dialog.getSaveFileName()


        if file_path is None: return
        
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        # Center the box
        dialog_geometry = file_dialog.geometry()
        screen_geometry = QApplication.desktop().screenGeometry() 
        screen_center = screen_geometry.center()
        dialog_center = dialog_geometry.center()
        file_dialog.move(screen_center - dialog_center)
        
        if self.csv_file is None: 
            return  
        
        if not self.csv_file.lower().endswith(".csv"):
            self.csv_file += ".csv"

        try:
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                headers = []
                for column in range(self.table_widget.columnCount()):
                    headers.append(self.table_widget.horizontalHeaderItem(column).text())
                writer.writerow(headers)

                for row in range(self.table_widget.rowCount()):
                    row_data = []
                    for column in range(self.table_widget.columnCount()):
                        item = self.table_widget.item(row, column)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            self.messageBox("success")
        except Exception as error:
            self.messageBox(f"Failed to export data: {str(error)}")
    

    def messageBox(self, result):
        msg_box = QMessageBox(self)  

        if result == "success":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Success")
            msg_box.setText("Data exported successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
        else:
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(result)
            msg_box.setStandardButtons(QMessageBox.Ok)

        screen_geometry = QApplication.desktop().screenGeometry()  
        screen_center = screen_geometry.center()
        msg_box.move(screen_center - msg_box.rect().center()) 
        msg_box.exec_()
        
    def messageBoxTraining(self, result):
        msg_box = QMessageBox(self)  
        if result == "success":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Success")
            msg_box.setText("Training in progress ...")
            msg_box.setStandardButtons(QMessageBox.Ok)
            
        else:
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Failed to train model...")
            msg_box.setStandardButtons(QMessageBox.Ok)

        screen_geometry = QApplication.desktop().screenGeometry()  
        screen_center = screen_geometry.center()
        msg_box.move(screen_center - msg_box.rect().center()) 
        msg_box.exec_()