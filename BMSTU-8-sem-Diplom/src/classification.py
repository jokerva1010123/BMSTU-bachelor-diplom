import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def get_train_data(csv_filename: str, mor):
    pos_data = pd.read_csv(csv_filename)
    pos_text = list(pos_data['text'].values)
    pos_title = list(pos_data['topic'].values)

    pos_text_train, pos_text_test, pos_title_train, pos_title_test = train_test_split(
        pos_text, pos_title, test_size=0.2, random_state=42
    )

    return pos_text_train, pos_text_test, pos_title_train, pos_title_test

def svm_classification(pos_text_train, pos_text_test, pos_title_train, pos_title_test, kernel='rbf'):
    tfidf_vect = TfidfVectorizer()
    x_train = tfidf_vect.fit_transform(pos_text_train)
    x_test = tfidf_vect.transform(pos_text_test)

    clf_svc = SVC(C=2.0, kernel=kernel, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                  tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape='ovo', break_ties=False, random_state=None)
    clf_svc.fit(x_train, pos_title_train)
    
    y_pred = clf_svc.predict(x_test)

    training_score = clf_svc.score(x_train, pos_title_train)
    test_score = clf_svc.score(x_test, pos_title_test)
    test1_score = accuracy_score(y_pred, pos_title_test)
    f1 = f1_score(pos_title_test, y_pred, average='weighted')

    return clf_svc, tfidf_vect, test_score, test1_score, f1