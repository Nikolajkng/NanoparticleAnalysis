from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox, QApplication, 
    QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout
)
from PIL import ImageQt
import os
import csv
import threading
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from threading import Event
from functools import partial 
import numpy as np

from src.gui.ui.MainUI import Ui_MainWindow
from src.controller.Controller import Controller
from src.shared.Commands import Command
from src.gui.windows.TrainModelWindow import TrainModelWindow
from src.gui.windows.SetScaleWindow import SetScaleWindow
from src.shared.ModelConfig import ModelConfig
from src.gui.TableData import TableData
from src.shared.ModelTrainingStats import ModelTrainingStats
from PIL import Image
from PIL.ImageQt import ImageQt
from src.gui.windows.MessageBoxes import *
from src.model.PlottingTools import plot_loss
from src.shared.ParticleImage import ParticleImage
from src.gui.windows.MessageBoxes import *
from src.model.PlottingTools import plot_loss
from src.shared.Formatters import _truncate
from src.shared.ParticleImage import ParticleImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow, Ui_MainWindow):
    update_train_model_values_signal = QtCore.pyqtSignal(ModelTrainingStats)

    def __init__(self):
        super().__init__()
        self.MainWindow = QMainWindow()
        self.setupUi(self.MainWindow)
        self.controller = Controller()
        self.image_path = None
        self.image: ParticleImage = None
        self.segmented_image = None
        self.annotated_image = None
        self.pixmap_item_count = None
        self.csv_file = None
        self.file_path_image = None
        self.scale_is_selected = False
        self.scale_input_set = False
        self.show_annotated_image = True  
        self.graphicsView_scene = QGraphicsScene(self)
        self.graphicsView.setScene(self.graphicsView_scene)
        self.training_state = "not done"
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

        self.plot_segmentation_scene = QGraphicsScene(self)
        self.plot_segmentation.setScene(self.plot_segmentation_scene)
        self.plot_graph_layout = QVBoxLayout(self.plot_histogram)
        self.plot_histogram.setLayout(self.plot_graph_layout)


        # Other connections
        self.action_train_model.triggered.connect(self.on_train_model_clicked)
        self.action_test_model.triggered.connect(self.on_test_model_clicked)
        self.action_open_image.triggered.connect(self.on_open_image_clicked)
        self.actionRun_Segmentation_on_Current_Image.triggered.connect(self.on_segment_image_clicked)
        self.actionRun_Segmentation_on_folder.triggered.connect(self.on_segment_folder_clicked)
        self.action_load_model.triggered.connect(self.on_load_model_clicked)
        self.actionExport_Segmentation_2.triggered.connect(self.on_export_segmented_clicked)
        self.actionExport_Data_as_csv.triggered.connect(self.on_export_data_csv_clicked)
        self.action_new_data_train_model.triggered.connect(self.on_train_model_custom_data_clicked)
        self.fullscreen_image_button.clicked.connect(self.on_fullscreen_image_clicked)
        self.radioButton.toggled.connect(self.on_toggle_segmented_image_clicked)
        self.setScaleButton.clicked.connect(self.open_set_scale_window) 
        self.runSegmentationBtn.clicked.connect(self.on_segment_image_clicked)
    
    def open_set_scale_window(self):
        if (self.image_path == None or self.image == None):
            messageBox(self, "No image found")
            return
        self.set_scale_window = SetScaleWindow(self.image, self.file_path_image, overlay_updater=self)
        self.set_scale_window.show()
            
    def display_image_metadata_overlay(self, file_path: str):
        file_name = os.path.basename(file_path)

        image_width = str(self.image.file_info.width)
        image_height = str(self.image.file_info.height)
        image_real_width = str(_truncate(self.image.file_info.real_width, 2))
        image_real_height = str(_truncate(self.image.file_info.real_height, 2))

        metadata = (
            f"{image_real_width}x{image_real_height} {self.image.file_info.unit} "
            f"({image_width}x{image_height})"
        )

        html_text = f"""
        <table width="100%" cellspacing="0" cellpadding="0">
            <tr>
                <td align="left">{file_name}</td>
                <td align="right">{metadata}</td>
            </tr>
        </table>
        """

        self.input_image_metadata.setText(html_text)

    
    
    def toggle_count_overlay(self):
            print("toggling")
            if self.segmented_image is None or self.annotated_image is None:
                messageBox(self, "No segmented image available to toggle.")
                return

            self.show_annotated_image = not self.show_annotated_image
            self.update_segmented_image_view()
        
    def on_fullscreen_image_clicked(self):
        if (self.image_path == None):
            messageBox(self, "Fullscreen failed: No image found")
            return
        import matplotlib
        matplotlib.use('Qt5Agg') 
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        
        manager = plt.get_current_fig_manager()

        axes[0].imshow(self.image.pil_image, cmap='gray')
        axes[0].set_title("Image")


        if self.show_annotated_image and self.annotated_image is not None:
            axes[1].imshow(self.annotated_image, cmap='gray')
            axes[1].set_title("Annotated Segmentation")  
        elif self.segmented_image is not None:
            axes[1].imshow(self.segmented_image, cmap='gray')
            axes[1].set_title("Segmentation")

        manager.window.showMaximized()
        plt.pause(0.1)
        plt.tight_layout()
        #plt.show(block=False)
        #fig.tight_layout()
        #plt.show()

    def set_table_data(self, table_data: np.ndarray):
        data = TableData(table_data)
        data.insertIn(self.table_widget)

    def on_segment_folder_clicked(self):
        input_folder_path = QFileDialog.getExistingDirectory(None, "Select an input folder", "")
        output_folder_path = QFileDialog.getExistingDirectory(None, "Select a folder for the output", "")
        if input_folder_path and output_folder_path:
            self.controller.process_command(Command.SEGMENT_FOLDER, input_folder_path, output_folder_path)
        else:
            messageBox(self, "Error in uploading directory")
            return

    def on_train_model_custom_data_clicked(self):
        self.train_model_window = TrainModelWindow(self.update_train_model_values_signal)
        self.train_model_window.train_model_signal.connect(self.train_model_custom_data)
        self.train_model_window.show()


    def train_model_custom_data(self, model_config: ModelConfig, stop_training_event: Event):
        result = confirmTrainingMessageBox(self, "Training a new model may take a while, do you want to continue?")
        if result == QMessageBox.No:
                return
                    
        try:
            self.train_thread = threading.Thread(
                target=partial(
                    self.controller.process_command, 
                    Command.RETRAIN, model_config, 
                    stop_training_event,
                    self.update_training_model_stats),
                daemon=True)
            self.train_thread.start()
            # messageBoxTraining(self, "success")            
        except:
            messageBoxTraining(self, "")
        # iou, pixel_accuracy = self.controller.process_command(Command.RETRAIN, model_config, self.update_training_model_stats)
        # print(f"""Model IOU: {iou}\nModel Pixel Accuracy: {pixel_accuracy}""")


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
            self.segmented_image = None
            
        
        default_image_path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'images'))
        
        self.scale_is_selected = False
        self.scale_input_set = False
        
        self.file_path_image, _ = QFileDialog.getOpenFileName(
            self, 
            "Select an image file", 
            default_image_path, 
            "Image Files (*.png *.jpg *.jpeg *.tif *.dm3 *.dm4);;All Files (*)")
        
        self.image_path = self.file_path_image

        if self.file_path_image: 
            self.image = self.controller.process_command(Command.LOAD_IMAGE, self.file_path_image)
            if self.image.pil_image.width > 1024 or self.image.pil_image.height > 1024:
                self.image.resize((1024, 1024))
            pixmap = self.load_pixmap(self.image.pil_image)
            pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
            self.graphicsView_scene.clear()
            self.graphicsView_scene.addItem(pixmap_item)
            self.display_image_metadata_overlay(self.file_path_image)

    def load_pixmap(self, image: Image) -> QPixmap:
        qimage = ImageQt(image)
        return QPixmap.fromImage(qimage) 

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
        
        self.segmented_image, self.annotated_image, table_data, histogram_fig = self.controller.process_command(Command.SEGMENT, self.image, "data/statistics")
        self.set_table_data(table_data)
        self.update_segmented_image_view()
        self.display_histogram(histogram_fig)
        
    def display_histogram(self, histogram_fig):
        if histogram_fig is not None:
            if hasattr(self, 'histogram_canvas') and self.histogram_canvas:
                self.histogram_canvas.setParent(None)

            self.histogram_canvas = FigureCanvas(histogram_fig)
            self.plot_graph_layout.addWidget(self.histogram_canvas)
            self.histogram_canvas.draw()
            
        
    def on_toggle_segmented_image_clicked(self):
        if self.segmented_image is None or self.annotated_image is None:
            messageBox(self, "No segmented image available to toggle.")
            return

        self.show_annotated_image = not self.show_annotated_image
        self.update_segmented_image_view()

    def update_segmented_image_view(self):
        if self.show_annotated_image:
            image_to_display = self.annotated_image
        else:
            image_to_display = self.segmented_image
            
        image_temp = ImageQt(image_to_display)
        pixmap = QPixmap.fromImage(image_temp)
        pixmap_item = QGraphicsPixmapItem(pixmap.scaled(500, 500, aspectRatioMode=1))
        self.plot_segmentation_scene.clear()
        self.plot_segmentation_scene.addItem(pixmap_item)

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
                    None,
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
            "Select a model file", 
            default_models_path, 
            "PT Files (*.pt);;All Files (*)"
            )
        
        if file_path: 
            if "PT" in selected_filter:
                if not file_path.endswith(".pt"):  
                    file_path += ".pt"
                self.controller.process_command(Command.LOAD_MODEL, file_path)
                self.current_model_label.setText(f"{os.path.basename(file_path)}")
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
        
        if not file_path: 
            return

        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"


        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
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

