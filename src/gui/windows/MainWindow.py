import csv
import threading
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtCore import QRect, Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from gui.ui.MainUI import Ui_MainWindow
from controller.Controller import Controller
from shared.Commands import Command
from PIL import ImageQt
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QFileDialog, QMessageBox, QApplication, QGraphicsScene, QGraphicsPixmapItem, QRubberBand, QGraphicsLineItem, QGraphicsView
from functools import partial 
from gui.windows.SelectScaleWindow import SelectScaleWindow
from gui.windows.TrainModelWindow import TrainModelWindow
import numpy as np
from shared.ScaleInfo import ScaleInfo
from shared.ModelConfig import ModelConfig
from gui.TableData import TableData
from shared.ModelTrainingStats import ModelTrainingStats
class MainWindow(QMainWindow, Ui_MainWindow):
    update_train_model_values_signal = QtCore.pyqtSignal(ModelTrainingStats)

    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        self.controller = Controller()
        self.image_path = None
        self.segmented_image = None
        self.csv_file = None
        self.select_scale_window = None
        self.scale_start_x = 0
        self.scale_end_x = 0
        self.graphicsView_scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.graphicsView_scene)
        self.input_image_real_width = 0
        self.scale_info = None
        self.standard_model_config = ModelConfig(images_path="data/images",
                                                 masks_path="data/masks",
                                                 epochs=300,
                                                 learning_rate=0.0005,
                                                 with_early_stopping=True,
                                                 with_data_augmentation=True)
        
        self.train_model_window = None
        self.train_thread = None

        self.plot1_scene = QGraphicsScene(self)
        self.plot1.setScene(self.plot1_scene)
        self.plot2_scene = QGraphicsScene(self)
        self.plot2.setScene(self.plot2_scene)
        self.plot3_scene = QGraphicsScene(self)
        self.plot3.setScene(self.plot3_scene)

        self.action_train_model.triggered.connect(self.on_train_model_clicked)
        self.action_test_model.triggered.connect(self.on_test_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)
        self.actionExport_Data_as_csv.triggered.connect(self.on_export_data_csv_clicked)
        self.selectBarScaleButton.clicked.connect(self.on_select_bar_scale_clicked)
        self.action_new_data_train_model.triggered.connect(self.on_train_model_custom_data_clicked)
        
    def set_table_data(self, table_data: np.ndarray):
        data = TableData(table_data)
        data.insertIn(self.table_widget)

    def scale_bar_set_event(self, xcoords: list[int]):
        print(f"Recieved xcoords: [{xcoords[0]}, {xcoords[1]}]")
        scale_window_width = self.select_scale_window.size().width()
        graphics_view_width = self.graphicsView.size().width()
        self.scale_start_x, self.scale_end_x = xcoords[0] / (scale_window_width/graphics_view_width), xcoords[1] / (scale_window_width/graphics_view_width)

        self.on_calculate_input_image_size_clicked()
        print(f"{self.scale_start_x}, {self.scale_end_x}")

    def on_calculate_input_image_size_clicked(self):
        selected_scale_info = ScaleInfo(self.scale_start_x, 
                               self.scale_end_x, 
                               self.barScaleInputField.text(), 
                               self.graphicsView.size().width())

        self.scale_info = self.controller.process_command(Command.CALCULATE_REAL_IMAGE_WIDTH, selected_scale_info)

    def on_select_bar_scale_clicked(self):
        self.select_scale_window = SelectScaleWindow()
        pixmap = QPixmap(self.image_path) 
        pixmap_item = QGraphicsPixmapItem(pixmap.scaled(1024, 1024, aspectRatioMode=1))
        self.select_scale_window.image_scene.addItem(pixmap_item)

        self.select_scale_window.scale_bar_set_signal.connect(self.scale_bar_set_event)

        self.select_scale_window.show()

    def on_train_model_custom_data_clicked(self):
        self.train_model_window = TrainModelWindow(self.update_train_model_values_signal)
        self.train_model_window.train_model_signal.connect(self.train_model_custom_data)
        self.train_model_window.stop_training_signal.connect(self.stop_model_training)
        self.train_model_window.show()

    def train_model_custom_data(self, model_config: ModelConfig):

        try:
            self.train_thread = threading.Thread(
                target=partial(self.controller.process_command, Command.RETRAIN, model_config, self.update_training_model_stats),
                daemon=True)
            self.train_thread.start()
            
            self.messageBoxTraining("success")
        except:
            self.messageBoxTraining("")
        # iou, pixel_accuracy = self.controller.process_command(Command.RETRAIN, model_config, self.update_training_model_stats)
        # print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")

    def stop_model_training(self):
        return

    def update_training_model_stats(self, stats: ModelTrainingStats):
        self.update_train_model_values_signal.emit(stats)

    def on_open_image_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
        self.image_path = file_path
        if file_path: 
            pixmap = QPixmap(file_path) 
            pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
            self.graphicsView_scene.addItem(pixmap_item)

    def on_test_model_clicked(self):
        image_folder_path = QFileDialog.getExistingDirectory(None, "Select test images folder", "")
        mask_folder_path = QFileDialog.getExistingDirectory(None, "Select test masks folder", "")

        if image_folder_path and mask_folder_path:
            iou, pixel_accuracy = self.controller.process_command(Command.TEST_MODEL, image_folder_path, mask_folder_path)
            print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")

    def on_segment_image_clicked(self):
        if (self.image_path == None):
            self.messageBox("Segmentation failed: No image found")
            return
        self.segmented_image, table_data = self.controller.process_command(Command.SEGMENT, self.image_path, self.scale_info)
        self.set_table_data(table_data)
        segmented_image_temp = ImageQt.ImageQt(self.segmented_image)
        pixmap = QPixmap.fromImage(segmented_image_temp)
        pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
        self.plot3_scene.addItem(pixmap_item)


    def on_train_model_clicked(self):
        try:
            self.messageBoxTraining("success")
            iou, pixel_accuracy = self.controller.process_command(Command.RETRAIN, self.standard_model_config)
            print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")
        # TODO: Separat thread fucker live-plot op for hold-out op.
        # try:
        #     train_thread = threading.Thread(
        #         target=partial(self.controller.process_command, Command.RETRAIN, self.standard_model_config),
        #         daemon=True)
        #     train_thread.start()
            
        except:
            self.messageBoxTraining("")


    def on_load_model_clicked(self):
        
        file_path, selected_filter = QFileDialog.getOpenFileName(None, "Select a file", "", "PT Files (*.pt);;All Files (*)")
        if file_path: 
            if "PT" in selected_filter:
                file_path += ".pt"
                self.controller.process_command(Command.LOAD_MODEL, file_path)
                self.messageBox("success", "Model loaded successfully")
            else:
                self.messageBox("Error: The selected file is not a PT file.")
            
        


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
            self.messageBox("success", "Segmented image exported successfully")
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

            self.messageBox("success", "Data exported successfully")
        except Exception as error:
            self.messageBox(f"Failed to export data: {str(error)}")
    

    def messageBox(self, result, text=""):
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
            msg_box.setText("Training in progress...")
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