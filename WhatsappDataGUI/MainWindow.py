import os.path
import random

import pandas as pd

from GUI.MainWindow_ui import *
from pathlib import Path
from PyQt5.QtWidgets import QListWidgetItem, QMessageBox
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, QThread, QRunnable
import Preprocessing.Preprocessor as P
from Preprocessing.utils import addCustomLabel, eraseCustomLabel, addContext, eraseContext
import re

'''
class ParallelComputation(QRunnable):
    def __init__(self, list_items):
        super().__init__()
        self.list_items = list_items

    def run(self):
    '''


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.file_list = []
        self.current_file_id = 0
        self.data_folder = ''
        self.context_colors_dict = {'Empty': QColor(255,255,255)}
        self.save_directory = Path('GeneratedData/')
        self.save_directory.mkdir(exist_ok=True)



        self.PB_root_folder_select.clicked.connect(self.openDirectory)
        self.PB_next_file.clicked.connect(self.next_file)
        self.PB_previous_file.clicked.connect(self.previous_file)
        self.PB_erase_multimedia.clicked.connect(self.eraseMultimedia)
        self.PB_erase_removed_messages.clicked.connect(self.eraseRemovedMessages)
        self.PB_url2tag.clicked.connect(self.changeURL2Tag)
        self.PB_save_modifications.clicked.connect(self.saveModifications)
        self.PB_save_directory_select.clicked.connect(self.openSaveDirectory)
        self.PB_recover_contexts.clicked.connect(self.recoverContext)

        self.CB_add_context.stateChanged.connect(self.contextSetting)
        self.CB_custom_label.stateChanged.connect(self.customLabel)
        self.CB_view_completed_files.stateChanged.connect(self.viewCompletedFiles)
        self.CB_highlight_user.stateChanged.connect(self.highlightTargetUser)

        self.setLayout(self.gridLayout)
        #self.WL_utterance.itemSelectionChanged.connect(self.changeItemColor)

    def keyPressEvent(self, event):
        # Detectar si se presionó la tecla "Suprimir" o "Delete"
        if event.key() == Qt.Key_Delete:
            selected_item = self.WL_utterance.currentItem()  # Obtener el elemento seleccionado
            if selected_item:
                self.WL_utterance.takeItem(self.WL_utterance.row(selected_item))  # Eliminar el elemento

    def changeItemColor(self, add_context):
        selected_items = self.WL_utterance.selectedItems()

        for item in selected_items:
            if add_context:
                brush = QBrush(self.context_colors_dict[self.TE_context.toPlainText()])
                item.setBackground(brush)
            else:
                brush = QBrush(self.context_colors_dict['Empty'])  # Color Blanco
                item.setBackground(brush)


    def next_file(self):
        self.current_file_id = (self.current_file_id + 1) % len(self.file_list)
        self.WL_utterance.clear()
        self.CB_add_context.setCheckState(False)
        self.CB_custom_label.setCheckState(False)
        self.load_file()
        self.highlightTargetUser()

    def previous_file(self):
        self.current_file_id = (self.current_file_id - 1) % len(self.file_list)
        self.WL_utterance.clear()
        self.CB_add_context.setCheckState(False)
        self.CB_custom_label.setCheckState(False)
        self.load_file()
        self.highlightTargetUser()

    def openDirectory(self):
        if not os.path.exists(self.data_folder):
            self.data_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.LB_root_directory.setText(str(self.data_folder))
        self.file_list = [f for f in Path(self.LB_root_directory.text()).glob('*.txt')]
        self.load_file()

    def openSaveDirectory(self):
        path_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.LB_save_directory.setText(str(path_folder))

    def recoverContext(self):
        try:
            path_folder = QtWidgets.QFileDialog.getOpenFileName(self, 'Select File')

            data = pd.read_csv(str(Path(path_folder[0])), index_col=0)
            self.WL_utterance.clear()
            for index, content in data.iterrows():
                message_list = [message[1:] for message in content['conversation'].split(' <context>') if '|' in message]
                context = content['conversation'].split('|')[0]
                for message in message_list:
                    if context not in list(self.context_colors_dict.keys()):
                        self.context_colors_dict[context] = QColor(random.randint(0, 255),
                                                                   random.randint(0, 255),
                                                                   random.randint(0, 255))
                    brush = QBrush(self.context_colors_dict[context])
                    item = QListWidgetItem(f'{context}|{message}')
                    item.setBackground(brush)
                    self.WL_utterance.addItem(item)
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("File not selected")
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()




    def load_file(self):
        file_name = self.file_list[self.current_file_id]
        self.LB_selected_file.setText(f'[{self.current_file_id+1}/{len(self.file_list)}] UTTERANCE FILE: {file_name.name}')

        line_index = 0
        if file_name:
            with open(str(file_name), 'r', encoding='utf-8') as file:
                for line in file:
                    '''
                    If one message is in the form 'xxxxxxx\nxxxxxxxx\n' The second sequence is computed as empty as
                    we read file line by line and that second sequence would not have date and hour,
                    so a sanity check is done
                    '''
                    date_regexp = r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}'
                    if re.search(date_regexp, line.split('-')[0]) == None :
                        #line = f'{previous_info}: {line}'
                        last_line = self.WL_utterance.item(self.WL_utterance.count()-1)
                        last_line.setText(f'{last_line.text()} \n {line.strip()}')
                    else:
                        item = QListWidgetItem(P.removeExtraMetadata(line))
                        self.WL_utterance.addItem(item)

                    #previous_info = ':'.join(line.split(':')[0:2]) #Necessary for sanity check
                    line_index += 1

    def customLabel(self):
        num_elems = self.WL_utterance.count()
        if num_elems > 0:
            for index in range(num_elems):
                item = self.WL_utterance.item(index)
                if self.CB_custom_label.isChecked():
                   new_text = addCustomLabel(item.text())
                else:
                    new_text = eraseCustomLabel(item.text())
                item.setText(new_text)


    def contextSetting(self):
        sel_items = self.WL_utterance.selectedItems()



        for item in sel_items:
            if self.CB_add_context.isChecked():
                if self.TE_context.toPlainText() not in list(self.context_colors_dict.keys()):
                    self.context_colors_dict[self.TE_context.toPlainText()] = QColor(random.randint(0, 255),
                                                                                     random.randint(0, 255),
                                                                                     random.randint(0, 255)
                                                                                     )
                item.setText(addContext(self.TE_context.toPlainText(), item.text()))
                self.changeItemColor(True)
            else:
                item.setText(eraseContext(item.text()))
                self.changeItemColor(False)

    def eraseMultimedia(self):
        num_elems = self.WL_utterance.count()
        items_to_remove = []
        if num_elems > 0:
            for index in range(num_elems):
                item = self.WL_utterance.item(index)
                if '<Multimedia omitido>' in item.text():
                    items_to_remove.append(item)

            for item in items_to_remove:
                self.WL_utterance.takeItem(self.WL_utterance.row(item))


    def eraseRemovedMessages(self):
        num_elems = self.WL_utterance.count()
        items_to_remove = []
        if num_elems > 0:
            for index in range(num_elems):
                item = self.WL_utterance.item(index)
                if 'Se eliminó este mensaje.' in item.text():
                    items_to_remove.append(item)

            for item in items_to_remove:
                self.WL_utterance.takeItem(self.WL_utterance.row(item))

    def changeURL2Tag(self):
        num_elems = self.WL_utterance.count()
        regexp = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
        tag = '<url>'
        if num_elems > 0:
            for index in range(num_elems):
                item = self.WL_utterance.item(index)
                item.setText(re.sub(regexp, tag, item.text()))


    def saveModifications(self):
        data_dict = {'conversation': []}

        for context_color in list(self.context_colors_dict.keys()):
            elems = [self.WL_utterance.item(i).text() for i in range(self.WL_utterance.count())
                     if self.WL_utterance.item(i).background().color() == self.context_colors_dict[context_color]]
            if len(elems) > 0:
                data_dict['conversation'].append(' '.join(elems))

        data_storage = pd.DataFrame(data_dict)
        msg = QMessageBox()
        if os.path.exists(self.LB_save_directory.text()):
            data_storage.to_csv(str(Path(self.LB_save_directory.text()).joinpath(self.file_list[self.current_file_id].name).with_suffix('.csv')))
            msg.setWindowTitle("Save Successfully Completed")
            msg.setText(f"Modifications where successfully saved in "
                        f"{str(Path(self.LB_save_directory.text()).joinpath(self.file_list[self.current_file_id].name).with_suffix('.csv'))}")
            msg.setIcon(QMessageBox.Information)
            msg.exec_()
        else:

            msg.setWindowTitle("Error")
            msg.setText("Save Path is not valid or does not exist")
            msg.setIcon(QMessageBox.Critical)
            msg.exec_()


    def viewCompletedFiles(self):
        if self.CB_view_completed_files.isChecked():
            self.WL_utterance.clear()
            self.openDirectory()
        else:
            if os.path.exists(self.LB_save_directory.text()):
                self.saved_file_list = [f.name for f in Path(self.LB_save_directory.text()).glob('*.csv')]
                new_file_list = []
                for f in self.file_list:
                    if f.with_suffix('.csv').name not in self.saved_file_list:
                        new_file_list.append(f)
                    else:
                        data = pd.read_csv(str(Path(self.LB_save_directory.text()).joinpath(f.with_suffix('.csv').name)), index_col=0)
                        if data['conversation'].str.contains('Fill').any():
                            new_file_list.append(f)
                self.file_list = new_file_list
                self.WL_utterance.clear()
                self.current_file_id = 0
                self.load_file()


            else:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Save Path is not valid or does not exist")
                msg.setIcon(QMessageBox.Critical)
                msg.exec_()
                self.CB_view_completed_files.setCheckState(2)

    def highlightTargetUser(self):
        messages = [(self.WL_utterance.item(index).text(), index) for index in range(self.WL_utterance.count())]
        filtered_messages = [(target_message, target_message[1]) for target_message in messages if
                             f'{self.LE_target_user.text()}:' in target_message[0] and '<context>' not in
                             target_message[0]]
        if self.CB_highlight_user.isChecked():
            for target_message in filtered_messages:
                color = QBrush(QColor(255,0,0))
                self.WL_utterance.item(target_message[1]).setBackground(color)
        else:
            for target_message in filtered_messages:
                color = QBrush(QColor(255,255,255))
                self.WL_utterance.item(target_message[1]).setBackground(color)







if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()