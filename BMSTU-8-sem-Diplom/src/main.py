from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.uic import loadUi
import sys

import nltk
from pymorphy2 import MorphAnalyzer
import time

from dataset import make_dataset
from preprocessing import split_text, lemmatize_remove_stopwords
from classification import get_train_data, svm_classification

morph = MorphAnalyzer()
global clf_svc, tfidf_vect

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()

        loadUi("window.ui", self)

        self.train_btn.clicked.connect(self.train_model)
        self.use_btn.clicked.connect(self.find_topic)
        self.update_btn.clicked.connect(self.update_file)
        self.file_btn.clicked.connect(self.choose_file)
    
    def train_model(self):
        file = self.file_combo.currentText()
        file = 'data/' + file
        try:
            text_train, text_test, title_train, title_test = get_train_data(file, morph)
        except:
            QMessageBox.warning(self, "Ошибка", "Файл не существует!")
            return
        
        start = time.time()
        global clf_svc, tfidf_vect
        clf_svc, tfidf_vect, _, _, _ = svm_classification(
            text_train, text_test,
            title_train, title_test
        )
        QMessageBox.information(self, "Инфо", "Классификатор готов.")
        end = time.time()
        # print(end - start)

    def find_topic(self):
        try: 
            clf_svc, tfidf_vect
        except:
            QMessageBox.warning(self, "Ошибка", "Сначала необходимо\nобучить классификатор!\n")
            return
        text = self.input_text.toPlainText()
        #print(text)
        # text = "Эксперимент NASA проверит теорию относительности, Американские ученые в ближайшее время отправят на орбиту спутник, который проверит два фундаментальных предположения, выдвинутых Альбертом Эйнштейном в рамках общей теории относительности, сообщает Associated Press."
        split = split_text(text)
        lemma = lemmatize_remove_stopwords(split, morph)
        if len(split) < 1 or len(lemma) < 10:
            QMessageBox.warning(self, "Ошибка", "Введите текст еще раз!")
            return 
        #print(lemma)
        y_predic = tfidf_vect.transform([lemma])
        res = clf_svc.predict(y_predic)[0]
        self.res_text.setPlainText(res)

    def update_file(self):
        file = self.file_combo.currentText()
        st = file.find('_') + 1
        en = file.find('.')
        num = file[st:en]
        make_dataset(int(num)//5, morph)
        QMessageBox.information(self, "Инфо", "Файл обновлен")
    
    def choose_file(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', './', '(*.txt)')[0]
        if filename == '':
            return
        file = open(filename, 'r', encoding="utf-8")
        input_text  = file.read()
        self.input_text.setPlainText(input_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    app.exec_()