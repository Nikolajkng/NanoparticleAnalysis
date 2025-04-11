import csv
import threading
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from gui.ui.MainUI import Ui_MainWindow
from controller.Controller import Controller
from shared.Commands import Command
from PIL import ImageQt
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QFileDialog, QMessageBox, QApplication, QGraphicsScene, QGraphicsPixmapItem
from functools import partial 
from gui.windows.SelectScaleWindow import SelectScaleWindow
from gui.windows.TrainModelWindow import TrainModelWindow
import numpy as np
from shared.ScaleInfo import ScaleInfo
from shared.ModelConfig import ModelConfig
from gui.TableData import TableData
from shared.ModelTrainingStats import ModelTrainingStats
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QIntValidator
from gui.windows.MessageBoxes import *
from model.PlottingTools import plot_loss

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
        self.scale_is_selected = False
        self.scale_input_set = False
        self.graphicsView_scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.graphicsView_scene)
        self.input_image_real_width = 0
        self.scale_info = None
        self.input_image_pixel_width = 0
        self.input_image_pixel_unit = "nm"
        self.training_state = "not done"
        self.validator = QIntValidator(0, 99999999, self)  
        self.standard_model_config = ModelConfig(images_path="data/images",
                                                 masks_path="data/masks",
                                                 epochs=5,
                                                 learning_rate=0.0005,
                                                 with_early_stopping=True,
                                                 with_data_augmentation=True)
        
        self.model_window = None
        self.train_thread = None
        self.training_loss_values = []
        self.validation_loss_values = []

        self.plot2_scene = QGraphicsScene(self)
        self.plot2.setScene(self.plot2_scene)
        self.plot3_scene = QGraphicsScene(self)
        self.plot3.setScene(self.plot3_scene)

        # Other connections
        self.action_train_model.triggered.connect(self.on_train_model_clicked)
        self.action_test_model.triggered.connect(self.on_test_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)
        self.actionExport_Data_as_csv.triggered.connect(self.on_export_data_csv_clicked)
        self.selectBarScaleButton.clicked.connect(self.on_select_bar_scale_clicked)
        self.action_new_data_train_model.triggered.connect(self.on_train_model_custom_data_clicked)
        self.fullscreen_image_button.clicked.connect(self.on_fullscreen_image_clicked)
        self.barScaleInputField.setValidator(self.validator)
        
    def on_fullscreen_image_clicked(self):
        if (self.image_path == None):
            messageBox(self, "Fullscreen failed: No image found")
            return

        image = Image.open(self.image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Image")

        if self.segmented_image:
            axes[1].imshow(self.segmented_image, cmap='gray')
            axes[1].set_title("Segmentation")

        plt.tight_layout()
        plt.show()


    def set_table_data(self, table_data: np.ndarray):
        data = TableData(table_data)
        data.insertIn(self.table_widget)

    def scale_bar_set_event(self, xcoords: list[int]):
        print(f"Recieved xcoords: [{xcoords[0]}, {xcoords[1]}]")
        scale_window_width = self.select_scale_window.size().width()
        graphics_view_width = self.graphicsView.size().width()
        self.scale_start_x, self.scale_end_x = xcoords[0] / (scale_window_width/graphics_view_width), xcoords[1] / (scale_window_width/graphics_view_width)

        print(f"{self.scale_start_x}, {self.scale_end_x}")
        self. scale_is_selected = True
        self.selectBarScaleButton.setStyleSheet("background-color: yellow; color: black;")
        self.selectBarScaleButton.setStyleSheet("")
        

    def on_calculate_input_image_size_clicked(self):
        selected_scale_info = ScaleInfo(self.scale_start_x, 
                               self.scale_end_x, 
                               self.barScaleInputField.text(), 
                               self.graphicsView.size().width())

        self.scale_info = self.controller.process_command(Command.CALCULATE_REAL_IMAGE_WIDTH, selected_scale_info)

    def on_select_bar_scale_clicked(self):
        if (self.image_path == None):
            messageBox(self, "No image found. Please upload an image first.")
            return
        
        if(self.barScaleInputField.text() == ""):
            messageBox(self, "Please enter length of the scale bar first")
            return
        
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
        result = confirmTrainingMessageBox(self, "Training a new model may take a while, do you want to continue?")
        if result == QMessageBox.No:
                return
                    
        try:
            self.train_thread = threading.Thread(
                target=partial(
                    self.controller.process_command, 
                    Command.RETRAIN, model_config, 
                    self.update_training_model_stats),
                daemon=True)
            self.train_thread.start()
            # messageBoxTraining(self, "success")            
        except:
            messageBoxTraining(self, "")
        # iou, pixel_accuracy = self.controller.process_command(Command.RETRAIN, model_config, self.update_training_model_stats)
        # print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")

    def stop_model_training(self):
        return

    def update_training_model_stats(self, stats: ModelTrainingStats):
        self.update_train_model_values_signal.emit(stats)

    def update_loss_values(self, stats: ModelTrainingStats):
        # Update the GUI with the training stats
        self.training_loss_values.append(stats.training_loss)
        self.validation_loss_values.append(stats.validation_loss)
        plot_loss(self.training_loss_values, self.validation_loss_values)
        
    def on_open_image_clicked(self):
        #Remove old item
        if (self.image_path):
            self.graphicsView_scene.clear()
            self.image_path = None
            self.scale_is_selected = False
            self.scale_input_set = False
            self.barScaleInputField.setText("")
            self.segmented_image = None
            
        
        default_image_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'images'))
        
        self.scale_is_selected = False
        self.scale_input_set = False
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select a file", 
            default_image_path, 
            "Image Files (*.png *.jpg *.jpeg *.tif *.dm3 *.dm4);;All Files (*)")
        
        self.image_path = file_path
        if file_path: 
            pixmap = self.load_pixmap(file_path)
            pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
            self.graphicsView_scene.clear()
            self.graphicsView_scene.addItem(pixmap_item)

    def load_pixmap(self, file_path):
        if self.is_dm_format(file_path):
            size_info, pil_image = self.controller.process_command(Command.GET_DM_IMAGE, file_path)
            pixel_size, pixel_unit = size_info

            self.input_image_real_width = float(pixel_size[1]*pil_image.width)
            self.input_image_pixel_width = pixel_size[1]
            self.input_image_pixel_unit = pixel_unit[1]
            qimage = ImageQt(pil_image)
            return QPixmap.fromImage(qimage)
        else:
            return QPixmap(file_path) 

    def is_dm_format(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension in [".dm3", ".dm4"]
    
    def on_test_model_clicked(self):
        
        image_folder_path = QFileDialog.getExistingDirectory(None, "Select test images folder", "")
        mask_folder_path = QFileDialog.getExistingDirectory(None, "Select test masks folder", "")

        if image_folder_path and mask_folder_path:
            iou, pixel_accuracy = self.controller.process_command(Command.TEST_MODEL, image_folder_path, mask_folder_path)
            print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")
        else:
            messageBox(self, "Error in uploading directories")
            return

    

    def on_segment_image_clicked(self):
        if (self.image_path == None):
            messageBox(self, "Segmentation failed: No image found")
            return
        
        if(not self.scale_is_selected):
            messageBox(self, "Please use the ''Select Bar Scale'' button to select the scale")
            return
        
        if(self.barScaleInputField.text() == ""):
            messageBox(self, "Please enter length of the scale bar first")
            return
        
        self.on_calculate_input_image_size_clicked()
        
        self.segmented_image, table_data = self.controller.process_command(Command.SEGMENT, self.image_path, self.scale_info)
        self.set_table_data(table_data)
        segmented_image_temp = ImageQt(self.segmented_image)
        pixmap = QPixmap.fromImage(segmented_image_temp)
        pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
        self.plot3_scene.addItem(pixmap_item)


    def on_train_model_clicked(self):
        result = confirmTrainingMessageBox(self, "Training a new model may take a while, do you want to continue?")
        if not result:
            return

        try:
            self.train_thread = threading.Thread(
                target=partial(
                    self.controller.process_command,
                    Command.RETRAIN,
                    self.standard_model_config,
                    self.update_loss_values 
                ),
                daemon=True
            )
            self.train_thread.start()
            #messageBoxTraining(self, True)
        except Exception as e:
            messageBoxTraining(self, False)
            print(f"Error during training: {e}")
            


    def on_load_model_clicked(self):        
        default_models_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'models'))
        file_path, selected_filter = QFileDialog.getOpenFileName(
            None, 
            "Select a file", 
            default_models_path, 
            "PT Files (*.pt);;All Files (*)"
            )
        
        if file_path: 
            if "PT" in selected_filter:
                if not file_path.endswith(".pt"):  
                    file_path += ".pt"
                self.controller.process_command(Command.LOAD_MODEL, file_path)
                messageBox(self, "success", "Model loaded successfully")
            else:
                messageBox(self, "Error: The selected file is not a PT file.")


    def on_export_segmented_clicked(self):
        if(self.segmented_image == None):
            messageBox(self, "Export failed: No segmented image was found to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            None, 
            "Save Image", 
            "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)")

        if file_path: 
            if not os.path.splitext(file_path)[1]: 
                if "PNG" in selected_filter:
                    file_path += ".png"
                elif "JPEG" in selected_filter:
                    file_path += ".jpg"
                else:
                    file_path += ".png" 
            
            self.segmented_image.save(file_path)            
            messageBox(self, "success", "Segmented image exported successfully")
        else:
            messageBox(self, "Error: File path is not selected.")
            
          
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

            messageBox(self, "success", "Data exported successfully")
        except Exception as error:
            messageBox(self, f"Failed to export data: {str(error)}")
    
