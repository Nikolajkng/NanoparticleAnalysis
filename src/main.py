import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.windows.MainWindow import MainWindow


app = QApplication(sys.argv)
ui = MainWindow()
ui.MainWindow.show()
sys.exit(app.exec_())