import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from src.gui.windows.MainWindow import MainWindow

def main():
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()