# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app_window.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1069, 848)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.controlTab = QtWidgets.QWidget()
        self.controlTab.setObjectName("controlTab")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.controlTab)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.detectionComboBox = QtWidgets.QComboBox(self.controlTab)
        self.detectionComboBox.setObjectName("detectionComboBox")
        self.detectionComboBox.addItem("")
        self.detectionComboBox.addItem("")
        self.gridLayout_3.addWidget(self.detectionComboBox, 2, 0, 1, 1)
        self.scaleComboBox = QtWidgets.QComboBox(self.controlTab)
        self.scaleComboBox.setObjectName("scaleComboBox")
        self.scaleComboBox.addItem("")
        self.gridLayout_3.addWidget(self.scaleComboBox, 6, 0, 1, 1)
        self.interpolationComboBox = QtWidgets.QComboBox(self.controlTab)
        self.interpolationComboBox.setObjectName("interpolationComboBox")
        self.interpolationComboBox.addItem("")
        self.interpolationComboBox.addItem("")
        self.interpolationComboBox.addItem("")
        self.gridLayout_3.addWidget(self.interpolationComboBox, 4, 0, 1, 1)
        self.interpolationLabel = QtWidgets.QLabel(self.controlTab)
        self.interpolationLabel.setWordWrap(True)
        self.interpolationLabel.setObjectName("interpolationLabel")
        self.gridLayout_3.addWidget(self.interpolationLabel, 3, 0, 1, 1)
        self.detectionAlgLabel = QtWidgets.QLabel(self.controlTab)
        self.detectionAlgLabel.setEnabled(True)
        self.detectionAlgLabel.setWordWrap(True)
        self.detectionAlgLabel.setObjectName("detectionAlgLabel")
        self.gridLayout_3.addWidget(self.detectionAlgLabel, 1, 0, 1, 1)
        self.inferenceButton = QtWidgets.QPushButton(self.controlTab)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons/geolocation.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.inferenceButton.setIcon(icon1)
        self.inferenceButton.setObjectName("inferenceButton")
        self.gridLayout_3.addWidget(self.inferenceButton, 0, 0, 1, 1)
        self.scaleAlgLabel = QtWidgets.QLabel(self.controlTab)
        self.scaleAlgLabel.setWordWrap(True)
        self.scaleAlgLabel.setObjectName("scaleAlgLabel")
        self.gridLayout_3.addWidget(self.scaleAlgLabel, 5, 0, 1, 1)
        self.calculateButton = QtWidgets.QPushButton(self.controlTab)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icons/calculator.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.calculateButton.setIcon(icon2)
        self.calculateButton.setObjectName("calculateButton")
        self.gridLayout_3.addWidget(self.calculateButton, 0, 1, 1, 1)
        self.clearButton = QtWidgets.QPushButton(self.controlTab)
        self.clearButton.setObjectName("clearButton")
        self.gridLayout_3.addWidget(self.clearButton, 6, 2, 1, 1)
        self.reportButton = QtWidgets.QPushButton(self.controlTab)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icons/report.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.reportButton.setIcon(icon3)
        self.reportButton.setObjectName("reportButton")
        self.gridLayout_3.addWidget(self.reportButton, 0, 2, 1, 1)
        self.stopButton = QtWidgets.QPushButton(self.controlTab)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icons/minus-circle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(icon4)
        self.stopButton.setObjectName("stopButton")
        self.gridLayout_3.addWidget(self.stopButton, 4, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.controlTab)
        self.textEdit.setReadOnly(True)
        self.textEdit.setAcceptRichText(False)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_4.addWidget(self.textEdit, 1, 0, 1, 1)
        self.tabWidget.addTab(self.controlTab, "")
        self.correctionToolTab = QtWidgets.QWidget()
        self.correctionToolTab.setObjectName("correctionToolTab")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.correctionToolTab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.previousButton = QtWidgets.QPushButton(self.correctionToolTab)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icons/arrow-180.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.previousButton.setIcon(icon5)
        self.previousButton.setObjectName("previousButton")
        self.gridLayout_5.addWidget(self.previousButton, 0, 0, 1, 1)
        self.openImageButton = QtWidgets.QPushButton(self.correctionToolTab)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icons/document.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openImageButton.setIcon(icon6)
        self.openImageButton.setObjectName("openImageButton")
        self.gridLayout_5.addWidget(self.openImageButton, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.correctionToolTab)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 2, 1, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.correctionToolTab)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_5.addWidget(self.graphicsView, 5, 0, 1, 2)
        self.nextButton = QtWidgets.QPushButton(self.correctionToolTab)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icons/arrow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.nextButton.setIcon(icon7)
        self.nextButton.setObjectName("nextButton")
        self.gridLayout_5.addWidget(self.nextButton, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.correctionToolTab)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 4, 2, 1, 1)
        self.imagesListWidget = QtWidgets.QListWidget(self.correctionToolTab)
        self.imagesListWidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imagesListWidget.sizePolicy().hasHeightForWidth())
        self.imagesListWidget.setSizePolicy(sizePolicy)
        self.imagesListWidget.setObjectName("imagesListWidget")
        self.gridLayout_5.addWidget(self.imagesListWidget, 5, 2, 1, 1)
        self.scaleSpinBox = QtWidgets.QSpinBox(self.correctionToolTab)
        self.scaleSpinBox.setMaximum(999999999)
        self.scaleSpinBox.setObjectName("scaleSpinBox")
        self.gridLayout_5.addWidget(self.scaleSpinBox, 4, 1, 1, 1)
        self.instanceButton = QtWidgets.QPushButton(self.correctionToolTab)
        self.instanceButton.setObjectName("instanceButton")
        self.gridLayout_5.addWidget(self.instanceButton, 4, 0, 1, 1)
        self.tabWidget.addTab(self.correctionToolTab, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1069, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setTearOffEnabled(False)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icons/question.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuHelp.setIcon(icon8)
        self.menuHelp.setToolTipsVisible(True)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_Dir = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icons/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen_Dir.setIcon(icon9)
        self.actionOpen_Dir.setObjectName("actionOpen_Dir")
        self.actionChange_Save_Dir = QtWidgets.QAction(MainWindow)
        self.actionChange_Save_Dir.setIcon(icon9)
        self.actionChange_Save_Dir.setObjectName("actionChange_Save_Dir")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("icons/disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon10)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setIcon(icon10)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionAppIcon = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("../../../../../../../../.designer/backup/icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAppIcon.setIcon(icon11)
        self.actionAppIcon.setObjectName("actionAppIcon")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.menuFile.addAction(self.actionOpen_Dir)
        self.menuFile.addAction(self.actionChange_Save_Dir)
        self.menuFile.addAction(self.actionSave)
        self.menuHelp.addAction(self.actionHelp)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TarMass"))
        self.detectionComboBox.setItemText(0, _translate("MainWindow", "YOLOv8m pose"))
        self.detectionComboBox.setItemText(1, _translate("MainWindow", "keypoint RCNN"))
        self.scaleComboBox.setItemText(0, _translate("MainWindow", "find-contours + heuristics"))
        self.interpolationComboBox.setItemText(0, _translate("MainWindow", "quadratic"))
        self.interpolationComboBox.setItemText(1, _translate("MainWindow", "cubic"))
        self.interpolationComboBox.setItemText(2, _translate("MainWindow", "slinear"))
        self.interpolationLabel.setText(_translate("MainWindow", "Line interpolation algorithm"))
        self.detectionAlgLabel.setText(_translate("MainWindow", "Keypoints detecion algorithm"))
        self.inferenceButton.setText(_translate("MainWindow", "Inference"))
        self.scaleAlgLabel.setText(_translate("MainWindow", "Scale detection algorithm"))
        self.calculateButton.setText(_translate("MainWindow", "Measure"))
        self.clearButton.setText(_translate("MainWindow", "Clear"))
        self.reportButton.setText(_translate("MainWindow", "Generate report"))
        self.stopButton.setText(_translate("MainWindow", "Stop process"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.controlTab), _translate("MainWindow", "Control"))
        self.previousButton.setText(_translate("MainWindow", "Previous"))
        self.openImageButton.setText(_translate("MainWindow", "Open selcted image"))
        self.label.setText(_translate("MainWindow", "Scale value [ µm ]"))
        self.nextButton.setText(_translate("MainWindow", "Next"))
        self.label_4.setText(_translate("MainWindow", "File list"))
        self.instanceButton.setText(_translate("MainWindow", "Create instance"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.correctionToolTab), _translate("MainWindow", "Correction tool"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_Dir.setText(_translate("MainWindow", "Open Dir"))
        self.actionChange_Save_Dir.setText(_translate("MainWindow", "Change Save Dir"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionAppIcon.setText(_translate("MainWindow", "AppIcon"))
        self.actionHelp.setText(_translate("MainWindow", "User Manual"))
