from PyQt5.QtWidgets import QMessageBox, QApplication

def confirmTrainingMessageBox(parent, message):
    confirmation = QMessageBox(parent)
    confirmation.setIcon(QMessageBox.Question)
    confirmation.setWindowTitle("Confirmation")
    confirmation.setText(message)
    confirmation.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    confirmation.adjustSize()
    
    screen = QApplication.screenAt(parent.geometry().center()) 
    if screen:
        screen_geometry = screen.geometry()  
        screen_center = screen_geometry.center()  
        confirmation.move(screen_center - confirmation.rect().center())
    return confirmation.exec_()



from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import QRect

def messageBox(parent, result, text=""):
    msg_box = QMessageBox(parent)

    if result == "success":
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText(text)
    else:
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(result)

    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.adjustSize()
    screen = QApplication.screenAt(parent.geometry().center()) 
    if screen:
        screen_geometry = screen.geometry() 
        screen_center = screen_geometry.center() 
        msg_box.move(screen_center - msg_box.rect().center())
    msg_box.exec_()



def messageBoxTraining(parent, is_success):
    msg_box = QMessageBox(parent)
    if is_success:
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Training completed successfully!")
    else:
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText("Failed to train model...")

    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.adjustSize()
    screen = QApplication.screenAt(parent.geometry().center()) 
    if screen:
        screen_geometry = screen.geometry() 
        screen_center = screen_geometry.center() 
        msg_box.move(screen_center - msg_box.rect().center())
    msg_box.exec_()
