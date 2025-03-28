from PyQt5 import QtCore, QtGui, QtWidgets
from gui.ui.TrainModelUI import Ui_TrainModel
from PyQt5.QtWidgets import QFileDialog, QMainWindow  
from shared.ModelConfig import ModelConfig
from shared.ModelTrainingStats import ModelTrainingStats
class TrainModelWindow(QMainWindow, Ui_TrainModel):
    train_model_signal = QtCore.pyqtSignal(ModelConfig)
    stop_training_signal = QtCore.pyqtSignal()
    def __init__(self, update_data_signal):
        super().__init__()
        self.setupUi(self)

        self.training_images_directory = None
        self.training_labels_directory = None
        self.test_images_directory = None
        self.test_labels_directory = None
        
        update_data_signal.connect(self.update_loss_values)

        self.training_images_button.clicked.connect(self.select_training_images_clicked)
        self.training_labels_button.clicked.connect(self.select_training_labels_clicked)
        self.test_images_button.clicked.connect(self.select_test_images_clicked)
        self.test_labels_button.clicked.connect(self.select_test_labels_clicked)
        self.train_model_button.clicked.connect(self.train_model_clicked)
        self.stop_training_button.clicked.connect(self.stop_training_clicked)

        self.auto_test_set_checkbox.stateChanged.connect(self.auto_test_set_checkbox_clicked)


    def open_directory(self, window_text):
        folder_path = QFileDialog.getExistingDirectory(None, window_text, "")
        if folder_path:
            return folder_path
        return
    
    def select_training_images_clicked(self):
        folder_path = self.open_directory("Select training images folder")
        if folder_path:
            self.training_images_directory = folder_path
    
    def select_training_labels_clicked(self):
        folder_path = self.open_directory("Select training labels folder")
        if folder_path:
            self.training_labels_directory = folder_path

    def select_test_images_clicked(self):
        folder_path = self.open_directory("Select test images folder")
        if folder_path:
            self.test_images_directory = folder_path

    def select_test_labels_clicked(self):
        folder_path = self.open_directory("Select test labels folder")
        if folder_path:
            self.test_labels_directory = folder_path
    
    def auto_test_set_checkbox_clicked(self):
        if self.auto_test_set_checkbox.isChecked():
            self.test_images_button.setEnabled(False)
            self.test_labels_button.setEnabled(False)
        else:
            self.test_images_button.setEnabled(True)
            self.test_labels_button.setEnabled(True)

    def train_model_clicked(self):
        self.stop_training_button.setEnabled(True)
        self.train_model_button.setEnabled(False)
        model_config = ModelConfig(images_path=self.training_images_directory,
                    masks_path=self.training_labels_directory,
                    epochs=int(self.epochs_input.text()),
                    learning_rate=float(self.learning_rate_input.text()),
                    with_early_stopping=self.early_stopping_checkbox.isChecked(),
                    with_data_augmentation=self.data_augment_checkbox.isChecked())
        print(int(self.epochs_input.text()))
        self.train_model_signal.emit(model_config)
    
    def stop_training_clicked(self):
        self.stop_training_button.setEnabled(False)
        self.train_model_button.setEnabled(True)
        self.stop_training_signal.emit()

    
    def update_loss_values(self, stats: ModelTrainingStats):
        self.training_loss_data_label.setText(str(stats.training_loss))
        self.val_loss_data_label.setText(str(stats.validation_loss))
        self.best_val_loss_data_label.setText(str(stats.best_loss))
        self.current_epoch_data_label.setText(str(stats.epoch))
        self.best_epoch_data_label.setText(str(stats.best_epoch))
