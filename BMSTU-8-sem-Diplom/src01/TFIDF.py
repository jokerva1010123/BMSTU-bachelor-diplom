import numpy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval

def text_preprocessing(text):
    # Step - 1b : Chuyển toàn bộ văn bản sang chữ thường
    text = str(text)
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    # # Step - 1c : Tokenization
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
    

def fit(corpus):
    unique_words = set()
    for doc in corpus:
        doc = literal_eval(doc)
        for word in doc:
            unique_words.add(word)
    unique_words = sorted(list(unique_words))
    vocab = {j:i for i,j in enumerate(unique_words)}
    return vocab
def IDF(corpus, vocab):
    doc_dict = {word:0 for word in vocab.keys()}
    for word in doc_dict.keys():
        for doc in corpus:
            if word in doc.split(" "):
                doc_dict[word] += 1
    IDF = 1 + numpy.log([(len(corpus) + 1) / (1 + frequency) for frequency in doc_dict.values()])
    return IDF
def transform(corpus, vocab, IDF):
    row_index = []
    column_index = []
    values = []
    for index, doc in enumerate(corpus):
        bow = doc.split(" ")
        boww = set(bow)
        for word in boww:
            if word in vocab.keys():
                word_count = bow.count(word)
                tf = word_count
                row_index.append(index)
                column_index.append(vocab[word])
                values.append(tf*IDF[column_index[-1]])
    tf_idf = csr_matrix((values, (row_index, column_index)), shape = (len(corpus), len(vocab)))      
    return normalize(tf_idf)

Corpus = pd.read_csv(
    r"D:/BMSTU/BMSTU-8-sem-Diplom/src/3/BBCNewsTrain2.csv",
    encoding="latin-1",
)
Corpus["text"].dropna(inplace=True)
tag_map = defaultdict(lambda: wn.NOUN)
tag_map["J"] = wn.ADJ
tag_map["V"] = wn.VERB
tag_map["R"] = wn.ADV

Corpus["text_final"] = Corpus["text"].map(text_preprocessing)
# print(Corpus["text"])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    Corpus["text_final"], Corpus["label"], test_size=0.5
)

Encoder = LabelEncoder()
Encoder.fit(Train_Y)
Train_Y = Encoder.transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)

# corpus = [
#     'this is the first document',
#     'this document is the second document'
# ]

vocab = fit(list(Corpus['text_final']))
# IDF = IDF(Corpus['text_final'], vocab)
# TF_IDF = transform(Train_X, vocab, IDF)


Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['text_final'])
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)

# print(Corpus["text_final"][0])
# print(vocab)
print(len(vocab))
print(len(Tfidf_vect.vocabulary_))
# print(vocab == sorted(Tfidf_vect.vocabulary_))