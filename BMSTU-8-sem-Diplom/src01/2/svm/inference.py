import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pickle
import re
# WordNetLemmatizer : Xác định danh , tính ,động từ
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Chuyển toàn bộ văn bản sang chữ thường
    text = str(text)
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Step - 1c : Tokenization
    text_words_list = word_tokenize(text)

    # Step - 1d : Làm sạch từ
    # Khai báo các biến để lưu
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # Kiểm tra danh tính động từ
    for word, tag in pos_tag(text_words_list):
        # Lưu ý chỉ xem xét bảng chữ cái English là alpha
        if word not in stopwords.words("english") and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    return str(Final_words)


# Loading Label encoder
labelencode = pickle.load(open('D:/BMSTU/BMSTU-8-sem-Diplom/src/2/svm/labelencoder_fitted.pkl', 'rb'))

# Loading TF-IDF Vectorizer
Tfidf_vect = pickle.load(open('D:/BMSTU/BMSTU-8-sem-Diplom/src/2/svm/Tfidf_vect_fitted.pkl', 'rb'))

# Loading models
SVM = pickle.load(open('D:/BMSTU/BMSTU-8-sem-Diplom/src/2/svm/svm_trained_model.sav', 'rb'))
Naive = pickle.load(open('D:/BMSTU/BMSTU-8-sem-Diplom/src/2/svm/nb_trained_model.sav', 'rb'))


# Inference
# sample_text = "The best soundtrack ever to anything"
# sample_text_processed = text_preprocessing(sample_text)
# sample_text_processed_vectorized = Tfidf_vect.transform([sample_text_processed])


# prediction_SVM = SVM.predict(sample_text_processed_vectorized)
# prediction_Naive = Naive.predict(sample_text_processed_vectorized)

# print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])
# print("Prediction from NB Model:", labelencode.inverse_transform(prediction_Naive)[0])


# API Text Classifier
def textClassifier(content):
    text_preprocessing(content)
    text_processed = text_preprocessing(content)
    text_processed_vectorized = Tfidf_vect.transform([text_processed])
    prediction_SVM = SVM.predict(text_processed_vectorized)
    return labelencode.inverse_transform(prediction_SVM)[0]
