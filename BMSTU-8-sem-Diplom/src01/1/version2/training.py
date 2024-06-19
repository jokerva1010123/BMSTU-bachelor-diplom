import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
import pickle
import nltk

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
# Set Random
np.random.seed(500)

# Thêm đường dẫn pandas

Corpus = pd.read_csv(
    r"/Users/dangnguyen/Documents/Project/Text-Classification/corpus.csv",
    encoding="latin-1",
)

# Step - 1: Tiền xử lý dữ liệu

# Step - 1a : Xoá các hàng trống
Corpus["text"].dropna(inplace=True)

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map["J"] = wn.ADJ
tag_map["V"] = wn.VERB
tag_map["R"] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words("english") and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


Corpus["text_final"] = Corpus["text"].map(text_preprocessing)

# Step - 2: Split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    Corpus["text_final"], Corpus["label"], test_size=0.3
)

# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Encoder.fit(Train_Y)
Train_Y = Encoder.transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus["text_final"])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Step - 5: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)


# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Dùng hàm accuracy_score function to get the accuracy
# ConfusionMatrixDisplay(confusion_matrix(Test_Y,predictions_NB),display_labels=['__label__1','__label__2']).plot()
print("Naive Bayes Accuracy Score -> ",
      accuracy_score(predictions_NB, Test_Y) * 100)

# Classifier - Algorithm - SVM
# Huấn luyện bộ dữ liệu trên phân loại
SVM = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")
SVM.fit(Train_X_Tfidf, Train_Y)

# Gán nhãn => predict
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Accuracy_score => accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)

# Saving Encdoer, TFIDF Vectorizer

# Lưu bộ mã hoá
filename = "labelencoder_fitted.pkl"
pickle.dump(Encoder, open(filename, "wb"))

# Lưu TFIDF Vectorizer
filename = "Tfidf_vect_fitted.pkl"
pickle.dump(Tfidf_vect, open(filename, "wb"))

# Lưu mô hình
filename = "svm_trained_model.sav"
pickle.dump(SVM, open(filename, "wb"))

filename = "nb_trained_model.sav"
pickle.dump(Naive, open(filename, "wb"))

print("Files saved to disk! Proceed to inference.py")
