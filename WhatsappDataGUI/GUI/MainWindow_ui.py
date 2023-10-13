# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1121, 847)
        MainWindow.setStyleSheet("background-color: rgb(37,211,102);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(20, -1, 0, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: White;")
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.PB_url2tag = QtWidgets.QPushButton(self.centralwidget)
        self.PB_url2tag.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_url2tag.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_url2tag.setFont(font)
        self.PB_url2tag.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_url2tag.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_url2tag.setObjectName("PB_url2tag")
        self.verticalLayout.addWidget(self.PB_url2tag)
        self.PB_erase_removed_messages = QtWidgets.QPushButton(self.centralwidget)
        self.PB_erase_removed_messages.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_erase_removed_messages.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_erase_removed_messages.setFont(font)
        self.PB_erase_removed_messages.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_erase_removed_messages.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_erase_removed_messages.setObjectName("PB_erase_removed_messages")
        self.verticalLayout.addWidget(self.PB_erase_removed_messages)
        self.PB_erase_multimedia = QtWidgets.QPushButton(self.centralwidget)
        self.PB_erase_multimedia.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_erase_multimedia.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_erase_multimedia.setFont(font)
        self.PB_erase_multimedia.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_erase_multimedia.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_erase_multimedia.setObjectName("PB_erase_multimedia")
        self.verticalLayout.addWidget(self.PB_erase_multimedia)
        self.CB_add_context = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.CB_add_context.setFont(font)
        self.CB_add_context.setStyleSheet("color: White;")
        self.CB_add_context.setObjectName("CB_add_context")
        self.verticalLayout.addWidget(self.CB_add_context)
        self.TE_context = QtWidgets.QTextEdit(self.centralwidget)
        self.TE_context.setMaximumSize(QtCore.QSize(300, 100))
        self.TE_context.setStyleSheet("background-color: rgb(255,255,255);\n"
"border: 2px  solid  rgb(255, 255, 255);\n"
"border-radius: 20px;\n"
"border-color: rgb(0,0,0);")
        self.TE_context.setObjectName("TE_context")
        self.verticalLayout.addWidget(self.TE_context)
        self.CB_custom_label = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.CB_custom_label.setFont(font)
        self.CB_custom_label.setStyleSheet("color: White;")
        self.CB_custom_label.setObjectName("CB_custom_label")
        self.verticalLayout.addWidget(self.CB_custom_label)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.COB_save_options = QtWidgets.QComboBox(self.centralwidget)
        self.COB_save_options.setStyleSheet("background-color: White;\n"
"border-color: Black;\n"
"border-radius: 10px;")
        self.COB_save_options.setObjectName("COB_save_options")
        self.COB_save_options.addItem("")
        self.horizontalLayout_5.addWidget(self.COB_save_options)
        self.PB_save_modifications = QtWidgets.QPushButton(self.centralwidget)
        self.PB_save_modifications.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_save_modifications.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_save_modifications.setFont(font)
        self.PB_save_modifications.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_save_modifications.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_save_modifications.setObjectName("PB_save_modifications")
        self.horizontalLayout_5.addWidget(self.PB_save_modifications)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.gridLayout.addLayout(self.verticalLayout, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PB_root_folder_select = QtWidgets.QPushButton(self.centralwidget)
        self.PB_root_folder_select.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PB_root_folder_select.setFont(font)
        self.PB_root_folder_select.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_root_folder_select.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_root_folder_select.setObjectName("PB_root_folder_select")
        self.horizontalLayout_2.addWidget(self.PB_root_folder_select)
        self.LB_root_directory = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LB_root_directory.setFont(font)
        self.LB_root_directory.setStyleSheet("background-color: rgb(255,255,255);\n"
"border: 2px  solid  rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"border-color: rgb(0,0,0);")
        self.LB_root_directory.setObjectName("LB_root_directory")
        self.horizontalLayout_2.addWidget(self.LB_root_directory)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.VL_utterance = QtWidgets.QVBoxLayout()
        self.VL_utterance.setObjectName("VL_utterance")
        self.LB_selected_file = QtWidgets.QLabel(self.centralwidget)
        self.LB_selected_file.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LB_selected_file.setFont(font)
        self.LB_selected_file.setStyleSheet("color: rgb(255,255,255)")
        self.LB_selected_file.setAlignment(QtCore.Qt.AlignCenter)
        self.LB_selected_file.setWordWrap(True)
        self.LB_selected_file.setObjectName("LB_selected_file")
        self.VL_utterance.addWidget(self.LB_selected_file)
        self.CB_view_completed_files = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.CB_view_completed_files.setFont(font)
        self.CB_view_completed_files.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.CB_view_completed_files.setAutoFillBackground(False)
        self.CB_view_completed_files.setStyleSheet("color: White;")
        self.CB_view_completed_files.setChecked(True)
        self.CB_view_completed_files.setObjectName("CB_view_completed_files")
        self.VL_utterance.addWidget(self.CB_view_completed_files)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(-1, 20, -1, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.LE_target_user = QtWidgets.QLineEdit(self.centralwidget)
        self.LE_target_user.setStyleSheet("background-color: rgb(255,255,255);\n"
"border: 2px  solid  rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"border-color: rgb(0,0,0);")
        self.LE_target_user.setObjectName("LE_target_user")
        self.horizontalLayout_6.addWidget(self.LE_target_user)
        self.CB_highlight_user = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.CB_highlight_user.setFont(font)
        self.CB_highlight_user.setStyleSheet("color: White;")
        self.CB_highlight_user.setObjectName("CB_highlight_user")
        self.horizontalLayout_6.addWidget(self.CB_highlight_user)
        self.VL_utterance.addLayout(self.horizontalLayout_6)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 40, -1, 20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.PB_previous_file = QtWidgets.QPushButton(self.centralwidget)
        self.PB_previous_file.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_previous_file.setMaximumSize(QtCore.QSize(200, 200))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_previous_file.setFont(font)
        self.PB_previous_file.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_previous_file.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_previous_file.setObjectName("PB_previous_file")
        self.horizontalLayout.addWidget(self.PB_previous_file)
        self.PB_recover_contexts = QtWidgets.QPushButton(self.centralwidget)
        self.PB_recover_contexts.setMaximumSize(QtCore.QSize(200, 200))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_recover_contexts.setFont(font)
        self.PB_recover_contexts.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_recover_contexts.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_recover_contexts.setObjectName("PB_recover_contexts")
        self.horizontalLayout.addWidget(self.PB_recover_contexts)
        self.PB_next_file = QtWidgets.QPushButton(self.centralwidget)
        self.PB_next_file.setMinimumSize(QtCore.QSize(0, 40))
        self.PB_next_file.setMaximumSize(QtCore.QSize(200, 200))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.PB_next_file.setFont(font)
        self.PB_next_file.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_next_file.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_next_file.setObjectName("PB_next_file")
        self.horizontalLayout.addWidget(self.PB_next_file)
        self.VL_utterance.addLayout(self.horizontalLayout)
        self.WL_utterance = QtWidgets.QListWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.WL_utterance.setFont(font)
        self.WL_utterance.setStyleSheet("QListWidget{background-color: rgb(255,255,255);\n"
"border: 2px  solid  rgb(255, 255, 255);\n"
"border-radius: 20px;\n"
"border-color: rgb(0,0,0);\n"
"}\n"
"\n"
"")
        self.WL_utterance.setLineWidth(3)
        self.WL_utterance.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.WL_utterance.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.WL_utterance.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.WL_utterance.setMovement(QtWidgets.QListView.Free)
        self.WL_utterance.setObjectName("WL_utterance")
        self.VL_utterance.addWidget(self.WL_utterance)
        self.gridLayout.addLayout(self.VL_utterance, 3, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 2, 1, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.PB_save_directory_select = QtWidgets.QPushButton(self.centralwidget)
        self.PB_save_directory_select.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PB_save_directory_select.setFont(font)
        self.PB_save_directory_select.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.PB_save_directory_select.setStyleSheet("QPushButton\n"
"{background: rgb(18, 140, 126);\n"
"border: 5px  solid  rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;\n"
"border-radius: 10px;\n"
"color: white;}\n"
"\n"
"QPushButton:hover {\n"
"    color: #000;\n"
"    background: qradialgradient(\n"
"        cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,\n"
"        radius: 1.35, stop: 0 #fff, stop: 1 #fff\n"
"        );\n"
"    }")
        self.PB_save_directory_select.setObjectName("PB_save_directory_select")
        self.horizontalLayout_4.addWidget(self.PB_save_directory_select)
        self.LB_save_directory = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.LB_save_directory.setFont(font)
        self.LB_save_directory.setStyleSheet("background-color: rgb(255,255,255);\n"
"border: 2px  solid  rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"border-color: rgb(0,0,0);")
        self.LB_save_directory.setObjectName("LB_save_directory")
        self.horizontalLayout_4.addWidget(self.LB_save_directory)
        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem3, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 8, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(0, 20, 500, -1)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255,255,255)")
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.LB_title = QtWidgets.QLabel(self.centralwidget)
        self.LB_title.setMaximumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.LB_title.setFont(font)
        self.LB_title.setStyleSheet("color: rgb(255,255,255)")
        self.LB_title.setText("")
        self.LB_title.setPixmap(QtGui.QPixmap(":/whatsapp_logo/ee994-logo-whatsapp-png.png"))
        self.LB_title.setScaledContents(True)
        self.LB_title.setAlignment(QtCore.Qt.AlignCenter)
        self.LB_title.setWordWrap(False)
        self.LB_title.setObjectName("LB_title")
        self.horizontalLayout_3.addWidget(self.LB_title)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setKerning(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(255,255,255)")
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1121, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "PREPROCESSING UTILS"))
        self.PB_url2tag.setText(_translate("MainWindow", "URL To Tag"))
        self.PB_erase_removed_messages.setText(_translate("MainWindow", "Erase Removed Messages"))
        self.PB_erase_multimedia.setText(_translate("MainWindow", "Erase Multimedia Messages"))
        self.CB_add_context.setText(_translate("MainWindow", "Add Context"))
        self.CB_custom_label.setText(_translate("MainWindow", "Add custom label"))
        self.COB_save_options.setItemText(0, _translate("MainWindow", "Save in .CSV with columns: [\'Conversation\']"))
        self.PB_save_modifications.setText(_translate("MainWindow", "Save Modifications"))
        self.PB_root_folder_select.setText(_translate("MainWindow", "Select Root Directory"))
        self.LB_root_directory.setText(_translate("MainWindow", "Directory will be displayed when selected..."))
        self.LB_selected_file.setText(_translate("MainWindow", "UTTERANCE FILE: CURRENTLY EMPTY"))
        self.CB_view_completed_files.setText(_translate("MainWindow", "View Completed Files"))
        self.CB_highlight_user.setText(_translate("MainWindow", "Highlight target user"))
        self.PB_previous_file.setText(_translate("MainWindow", "Previous File"))
        self.PB_recover_contexts.setText(_translate("MainWindow", "Recover Contexts  From File"))
        self.PB_next_file.setText(_translate("MainWindow", "Next File"))
        self.PB_save_directory_select.setText(_translate("MainWindow", "Select Save Directory"))
        self.LB_save_directory.setText(_translate("MainWindow", "Directory will be displayed when selected..."))
        self.label.setText(_translate("MainWindow", "WhatsApp"))
        self.label_2.setText(_translate("MainWindow", "Utterance Preprocessing"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
