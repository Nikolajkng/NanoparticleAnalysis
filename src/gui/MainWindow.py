# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_prototype1.2.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1178, 790)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_view = QtWidgets.QLabel(self.centralwidget)
        self.image_view.setGeometry(QtCore.QRect(190, 30, 481, 501))
        self.image_view.setText("")
        self.image_view.setObjectName("image_view")
        self.options_widget = QtWidgets.QListWidget(self.centralwidget)
        self.options_widget.setGeometry(QtCore.QRect(10, 10, 171, 521))
        self.options_widget.setObjectName("options_widget")
        item = QtWidgets.QListWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.options_widget.addItem(item)
        self.table_widget = QtWidgets.QTableWidget(self.centralwidget)
        self.table_widget.setGeometry(QtCore.QRect(10, 540, 1151, 201))
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.setObjectName("table_widget")
        self.table_widget.setColumnCount(4)
        self.table_widget.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(1, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(2, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(2, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(3, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(3, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(4, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setItem(4, 3, item)
        self.plots_tab_view = QtWidgets.QTabWidget(self.centralwidget)
        self.plots_tab_view.setGeometry(QtCore.QRect(690, 10, 471, 531))
        self.plots_tab_view.setObjectName("plots_tab_view")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.plot1 = QtWidgets.QLabel(self.tab1)
        self.plot1.setGeometry(QtCore.QRect(10, 10, 451, 481))
        self.plot1.setText("")
        self.plot1.setPixmap(QtGui.QPixmap("../../../Pictures/Screenshots/Screenshot from 2025-02-26 11-56-07.png"))
        self.plot1.setObjectName("plot1")
        self.plots_tab_view.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.plot2 = QtWidgets.QLabel(self.tab2)
        self.plot2.setGeometry(QtCore.QRect(0, 0, 461, 491))
        self.plot2.setText("")
        self.plot2.setPixmap(QtGui.QPixmap("../../../Pictures/Screenshots/Screenshot from 2025-02-26 11-51-50.png"))
        self.plot2.setObjectName("plot2")
        self.plots_tab_view.addTab(self.tab2, "")
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.plot3 = QtWidgets.QLabel(self.tab3)
        self.plot3.setGeometry(QtCore.QRect(10, 10, 451, 481))
        self.plot3.setText("")
        self.plot3.setPixmap(QtGui.QPixmap("../../../Pictures/Screenshots/Screenshot from 2025-02-26 11-44-39.png"))
        self.plot3.setObjectName("plot3")
        self.plots_tab_view.addTab(self.tab3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1178, 24))
        self.menubar.setObjectName("menubar")
        self.menu_file = QtWidgets.QMenu(self.menubar)
        self.menu_file.setObjectName("menu_file")
        self.menu_edit = QtWidgets.QMenu(self.menubar)
        self.menu_edit.setObjectName("menu_edit")
        self.menu_model = QtWidgets.QMenu(self.menubar)
        self.menu_model.setObjectName("menu_model")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuRun = QtWidgets.QMenu(self.menubar)
        self.menuRun.setObjectName("menuRun")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.statusbar.setFont(font)
        self.statusbar.setAutoFillBackground(False)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_train_model = QtWidgets.QAction(MainWindow)
        self.action_train_model.setObjectName("action_train_model")
        self.action_load_model = QtWidgets.QAction(MainWindow)
        self.action_load_model.setObjectName("action_load_model")
        self.action_open_image = QtWidgets.QAction(MainWindow)
        self.action_open_image.setObjectName("action_open_image")
        self.actionExport_Segmentation = QtWidgets.QAction(MainWindow)
        self.actionExport_Segmentation.setObjectName("actionExport_Segmentation")
        self.actionExport_Segmentation_2 = QtWidgets.QAction(MainWindow)
        self.actionExport_Segmentation_2.setObjectName("actionExport_Segmentation_2")
        self.actionExport_Data_as_csv = QtWidgets.QAction(MainWindow)
        self.actionExport_Data_as_csv.setObjectName("actionExport_Data_as_csv")
        self.actionRun_Segmentation_on_Current_Image = QtWidgets.QAction(MainWindow)
        self.actionRun_Segmentation_on_Current_Image.setObjectName("actionRun_Segmentation_on_Current_Image")
        self.menu_file.addAction(self.action_open_image)
        self.menu_model.addAction(self.action_train_model)
        self.menu_model.addAction(self.action_load_model)
        self.menuTools.addAction(self.actionExport_Segmentation_2)
        self.menuTools.addAction(self.actionExport_Data_as_csv)
        self.menuRun.addAction(self.actionRun_Segmentation_on_Current_Image)
        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_edit.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menu_model.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        self.plots_tab_view.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.options_widget.isSortingEnabled()
        self.options_widget.setSortingEnabled(False)
        item = self.options_widget.item(0)
        item.setText(_translate("MainWindow", "Analyze"))
        self.options_widget.setSortingEnabled(__sortingEnabled)
        item = self.table_widget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Count"))
        item = self.table_widget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Density"))
        item = self.table_widget.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Diameter"))
        item = self.table_widget.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "Height"))
        item = self.table_widget.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "Area"))
        item = self.table_widget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Mean"))
        item = self.table_widget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Minimum"))
        item = self.table_widget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Maximum"))
        item = self.table_widget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Sigma"))
        __sortingEnabled = self.table_widget.isSortingEnabled()
        self.table_widget.setSortingEnabled(False)
        item = self.table_widget.item(0, 0)
        item.setText(_translate("MainWindow", "387"))
        item = self.table_widget.item(0, 1)
        item.setText(_translate("MainWindow", "387"))
        item = self.table_widget.item(0, 2)
        item.setText(_translate("MainWindow", "387"))
        item = self.table_widget.item(0, 3)
        item.setText(_translate("MainWindow", "1"))
        item = self.table_widget.item(1, 0)
        item.setText(_translate("MainWindow", "23"))
        item = self.table_widget.item(1, 1)
        item.setText(_translate("MainWindow", "23"))
        item = self.table_widget.item(1, 2)
        item.setText(_translate("MainWindow", "42"))
        item = self.table_widget.item(1, 3)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_widget.item(2, 0)
        item.setText(_translate("MainWindow", "30"))
        item = self.table_widget.item(2, 1)
        item.setText(_translate("MainWindow", "40"))
        item = self.table_widget.item(2, 2)
        item.setText(_translate("MainWindow", "50"))
        item = self.table_widget.item(2, 3)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_widget.item(3, 0)
        item.setText(_translate("MainWindow", "701"))
        item = self.table_widget.item(3, 1)
        item.setText(_translate("MainWindow", "670"))
        item = self.table_widget.item(3, 2)
        item.setText(_translate("MainWindow", "1232"))
        item = self.table_widget.item(3, 3)
        item.setText(_translate("MainWindow", "4"))
        item = self.table_widget.item(4, 0)
        item.setText(_translate("MainWindow", "2043"))
        item = self.table_widget.item(4, 1)
        item.setText(_translate("MainWindow", "2001"))
        item = self.table_widget.item(4, 2)
        item.setText(_translate("MainWindow", "3012"))
        item = self.table_widget.item(4, 3)
        item.setText(_translate("MainWindow", "5"))
        self.table_widget.setSortingEnabled(__sortingEnabled)
        self.plots_tab_view.setTabText(self.plots_tab_view.indexOf(self.tab1), _translate("MainWindow", "???"))
        self.plots_tab_view.setTabText(self.plots_tab_view.indexOf(self.tab2), _translate("MainWindow", "Graph"))
        self.plots_tab_view.setTabText(self.plots_tab_view.indexOf(self.tab3), _translate("MainWindow", "Labeled"))
        self.menu_file.setTitle(_translate("MainWindow", "File"))
        self.menu_edit.setTitle(_translate("MainWindow", "Edit"))
        self.menu_model.setTitle(_translate("MainWindow", "Model"))
        self.menuTools.setTitle(_translate("MainWindow", "Export"))
        self.menuRun.setTitle(_translate("MainWindow", "Analyze"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.action_train_model.setText(_translate("MainWindow", "Train model"))
        self.action_load_model.setText(_translate("MainWindow", "Load model"))
        self.action_open_image.setText(_translate("MainWindow", "Open Image"))
        self.actionExport_Segmentation.setText(_translate("MainWindow", "Export Segmentation"))
        self.actionExport_Segmentation_2.setText(_translate("MainWindow", "Export Segmentation"))
        self.actionExport_Data_as_csv.setText(_translate("MainWindow", "Export Data as csv"))
        self.actionRun_Segmentation_on_Current_Image.setText(_translate("MainWindow", "Run Segmentation on Current Image"))
