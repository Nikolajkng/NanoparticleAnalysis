import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog, QLabel, 
    QVBoxLayout, QHBoxLayout, QWidget, QPushButton,QDesktopWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from AnalysisWindow import AnalysisWindow
from window_functions import centerWindow


# Linux stuff - supresses error
os.environ["XDG_SESSION_TYPE"] = "xcb"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();
        self.setWindowTitle("NanoParticleAnalysis v.1")
        self.setGeometry(0,0,1200,800)
        centerWindow(self)
        
 
        # Create main widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)  

        # Create Sidebar
        self.sidebar = self.createSidebar()
        self.main_layout.addWidget(self.sidebar, 1)  

        # Create Main Content Area (for image display)
        self.main_content = self.createMainContent()
        self.main_layout.addWidget(self.main_content, 3) 

         # Create top menubar
        self.createMenu()
       
       
    def createMainContent(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        self.image_label = QLabel("No image uploaded", self)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 40px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(self.image_label)
        return main_widget
       
    def createSidebar(self):
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)

        # Buttons
        upload_btn = QPushButton("Upload Image")
        upload_btn.clicked.connect(self.uploadFile)

        analyze_btn = QPushButton("Analyze")
        self.aWindow = AnalysisWindow()  
        analyze_btn.clicked.connect(self.aWindow.show)


        sidebar_layout.addStretch()
        sidebar_layout.addWidget(upload_btn)
        sidebar_layout.addWidget(analyze_btn)

        return sidebar_widget

    def createMenu(self):
        menubar = self.menuBar()

        # Add a menu entries
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        help_menu = menubar.addMenu('Help')


        # Create an 'Upload' action
        upload_action = QAction('Upload image', self)
        temp1_action = QAction('temp1', self)
        temp2_action = QAction('temp2', self)


        upload_action.triggered.connect(self.uploadFile) 
        
        # Add the action to the 'Menu Entry'
        file_menu.addAction(upload_action) 
        edit_menu.addAction(temp1_action)
        help_menu.addAction(temp2_action)
       
        
        
    def uploadFile(self):
            file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", "All Files (*)")
            
            if file_path: 
                pixmap = QPixmap(file_path) 
                self.image_label.setPixmap(pixmap.scaled(600, 600, aspectRatioMode=1))  

           


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()