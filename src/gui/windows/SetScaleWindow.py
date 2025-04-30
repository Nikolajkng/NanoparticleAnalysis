from PyQt5.QtWidgets import QDialog  
<<<<<<< Updated upstream

from src.gui.ui.SetScaleUI import Ui_set_scale_window

=======
from gui.ui.SetScaleUI import Ui_set_scale_window
from shared.Formatters import _truncate
from gui.windows.MessageBoxes import messageBox
from PyQt5.QtGui import QDoubleValidator
>>>>>>> Stashed changes

class SetScaleWindow(QDialog, Ui_set_scale_window):
    
    def __init__(self, uploaded_image):
        super().__init__()
        self.setupUi(self)
        self.image = uploaded_image
        self.file_info = self.image.file_info

        # Validators
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.StandardNotation)
        double_validator.setBottom(0)  

        # Values
        self.setScaleInputField_DistanceInPixels.setValidator(double_validator)
        self.setScaleInputField_KnownDistance.setValidator(double_validator)
        self.setScaleInputField_PixelRatio.setValidator(double_validator)
        self.setScaleInputField_DistanceInPixels.setText(str(self.file_info.width))
        self.setScaleInputField_KnownDistance.setText(self._get_knowndistance())
        self.setScaleInputField_PixelRatio.setText(self._calc_pixelratio())
        self.setScaleInputField_Unit.setText(self.file_info.unit)
        self.setScaleInputField_DistanceInPixels.textChanged.connect(self._update_scale_result)
        self.setScaleInputField_KnownDistance.textChanged.connect(self._update_scale_result)
        self.setScaleInputField_PixelRatio.textChanged.connect(self._update_scale_result)
        self.setScaleInputField_Unit.textChanged.connect(self._update_scale_result)


        self.label_scale_result.setText(self._calc_pixels_per_unit())

        
        # Buttons
        self.buttonBox.accepted.disconnect()
        self.buttonBox.accepted.connect(self.confirmBtnClicked) 
        self.buttonBox.rejected.disconnect()
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton.clicked.connect(self.clearAllValues)  


    def _get_knowndistance(self):
        return str(_truncate(self.file_info.real_width, 2))

    def _calc_pixelratio(self):
        result = self.file_info.pixel_width / self.file_info.pixel_height
        return str(_truncate(result, 2))

    def _calc_pixels_per_unit(self):
        result = self.file_info.width / self.file_info.real_width
        return str(_truncate(result, 2))

    def _update_scale_result(self):
        try:
            distance_in_pixels = float(self.setScaleInputField_DistanceInPixels.text())
            known_distance = float(self.setScaleInputField_KnownDistance.text())
            pixel_ratio = float(self.setScaleInputField_PixelRatio.text())
            unit = self.setScaleInputField_Unit.text()
            pixel_size = known_distance / distance_in_pixels
            pixels_per_unit = 1 / pixel_size
            
            self.setScaleInputField_DistanceInPixels.setStyleSheet("background-color: white;")
            self.setScaleInputField_KnownDistance.setStyleSheet("background-color: white;")
            self.setScaleInputField_PixelRatio.setStyleSheet("background-color: white;")
            self.setScaleInputField_Unit.setStyleSheet("background-color: white;")
            
            self.label_scale_result.setText(str(_truncate(pixels_per_unit, 2)))
        except ValueError:
            self.label_scale_result.setText("NaN")

    def confirmBtnClicked(self):
        try:
            if self.setScaleInputField_DistanceInPixels.text() == "":
                self.setScaleInputField_DistanceInPixels.setStyleSheet("background-color: red;")
                return
            
            if self.setScaleInputField_KnownDistance.text() == "":
                self.setScaleInputField_KnownDistance.setStyleSheet("background-color: red;")
                return
            if self.setScaleInputField_PixelRatio.text() == "":
                self.setScaleInputField_PixelRatio.setStyleSheet("background-color: red;")
                return

            distance_in_pixels = float(self.setScaleInputField_DistanceInPixels.text())
            known_distance = float(self.setScaleInputField_KnownDistance.text())
            pixel_ratio = float(self.setScaleInputField_PixelRatio.text())

            unit = self.setScaleInputField_Unit.text()
            pixel_size = known_distance / distance_in_pixels

            self.file_info.pixel_width = pixel_size
            self.file_info.pixel_height = pixel_size / pixel_ratio
            self.file_info.unit = unit

            self.accept()  

        except ValueError:
            self.label_scale_result.setText("NaN")
            return


    def clearAllValues(self):
        self.setScaleInputField_DistanceInPixels.clear()
        self.setScaleInputField_KnownDistance.clear()
        self.setScaleInputField_PixelRatio.clear()
        self.setScaleInputField_Unit.clear()
        self.label_scale_result.setText("")