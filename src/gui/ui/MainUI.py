# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui_prototype1.2_graphics_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1263, 797)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.options_widget = QtWidgets.QListWidget(self.centralwidget)
        self.options_widget.setGeometry(QtCore.QRect(10, 10, 221, 521))
        self.options_widget.setObjectName("options_widget")
        self.table_widget = QtWidgets.QTableWidget(self.centralwidget)
        self.table_widget.setGeometry(QtCore.QRect(10, 540, 1241, 201))
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
        self.plots_tab_view.setGeometry(QtCore.QRect(750, 10, 511, 531))
        self.plots_tab_view.setObjectName("plots_tab_view")
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.plot3 = QtWidgets.QGraphicsView(self.tab3)
        self.plot3.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.plot3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plot3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plot3.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.plot3.setObjectName("plot3")
        self.plots_tab_view.addTab(self.tab3, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.plot2 = QtWidgets.QGraphicsView(self.tab2)
        self.plot2.setGeometry(QtCore.QRect(0, 0, 500, 500))
        self.plot2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plot2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plot2.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.plot2.setObjectName("plot2")
        self.plots_tab_view.addTab(self.tab2, "")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(240, 30, 500, 500))
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.graphicsView.setObjectName("graphicsView")
        self.barScaleInputField = QtWidgets.QLineEdit(self.centralwidget)
        self.barScaleInputField.setGeometry(QtCore.QRect(30, 80, 131, 22))
        self.barScaleInputField.setInputMethodHints(QtCore.Qt.ImhNone)
        self.barScaleInputField.setText("")
        self.barScaleInputField.setObjectName("barScaleInputField")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(160, 80, 51, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.selectBarScaleButton = QtWidgets.QPushButton(self.centralwidget)
        self.selectBarScaleButton.setGeometry(QtCore.QRect(30, 50, 181, 24))
        self.selectBarScaleButton.setObjectName("selectBarScaleButton")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 30, 221, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 10, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.fullscreen_image_button = QtWidgets.QPushButton(self.centralwidget)
        self.fullscreen_image_button.setGeometry(QtCore.QRect(60, 480, 111, 31))
        self.fullscreen_image_button.setObjectName("fullscreen_image_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1263, 21))
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
        self.action_test_model = QtWidgets.QAction(MainWindow)
        self.action_test_model.setObjectName("action_test_model")
        self.action_new_data_train_model = QtWidgets.QAction(MainWindow)
        self.action_new_data_train_model.setObjectName("action_new_data_train_model")
        self.menu_file.addAction(self.action_open_image)
        self.menu_model.addAction(self.action_new_data_train_model)
        self.menu_model.addAction(self.action_load_model)
        self.menu_model.addAction(self.action_train_model)
        self.menu_model.addAction(self.action_test_model)
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
        self.table_widget.setSortingEnabled(__sortingEnabled)
        self.plots_tab_view.setTabText(self.plots_tab_view.indexOf(self.tab3), _translate("MainWindow", "Labeled"))
        self.plots_tab_view.setTabText(self.plots_tab_view.indexOf(self.tab2), _translate("MainWindow", "Graph"))
        self.barScaleInputField.setPlaceholderText(_translate("MainWindow", " Enter scale bar"))
        self.comboBox.setItemText(0, _translate("MainWindow", "nm"))
        self.comboBox.setItemText(1, _translate("MainWindow", "μm"))
        self.selectBarScaleButton.setText(_translate("MainWindow", "Select scale bar on image"))
        self.label.setText(_translate("MainWindow", "Analyze"))
        self.fullscreen_image_button.setText(_translate("MainWindow", "Fullscreen Image"))
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
        self.action_test_model.setText(_translate("MainWindow", "Test model"))
        self.action_new_data_train_model.setText(_translate("MainWindow", "Train a new model on custom data"))
