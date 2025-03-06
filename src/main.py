import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from gui.GUI import GUI


app = QApplication(sys.argv)
ui = GUI()
ui.MainWindow.show()
sys.exit(app.exec_())