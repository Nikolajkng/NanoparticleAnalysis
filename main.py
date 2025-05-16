import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from src.gui.windows.MainWindow import MainWindow

from src.model.CrossValidation import cv_kfold
def main():
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    #main()
    cv_kfold("data/medres_images", "data/medres_masks")