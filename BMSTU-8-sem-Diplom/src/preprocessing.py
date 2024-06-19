from typing import List
import string
from nltk import tokenize
from nltk.corpus import stopwords

def split_text(text: str, min_char: int = 1) -> List[str]:
    text = text.replace('.”', '”.').replace('."', '".').replace('?”', '”?').replace('!”', '”!')
    text = text.replace('--', ' ').replace('. . .', '').replace('_', '')
    text = text.replace('\xa0', '')

    sentences = tokenize.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) >= min_char]

    normed_text = []
    
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        sentence = sentence.replace('“', '').replace('”', '')
        sentence = sentence.replace('‟', '').replace('”', '')
        sentence = sentence.replace('«', '').replace('»', '')
        sentence = sentence.replace('-', '').replace('-', '')
        sentence = sentence.replace('(', '').replace(')', '')
        sentence = sentence.replace('…', '')
        sentence = sentence.replace('\n', '')
        normed_text.append(sentence)

    return list(normed_text)

def lemmatize(text: List[str], morph) -> List[str]:
    lemmatized_text = ""
    for sentence in text:
        tokens = tokenize.word_tokenize(sentence)
        lemmas = ''
        for token in tokens:
            lemma = morph.normal_forms(token)[0]
            lemmas += lemma + ' '
        lemmas = lemmas.rstrip()
        lemmatized_text += lemmas + " "
    lemmatized_text.rstrip()

    return lemmatized_text

def lemmatize_remove_stopwords(text: List[str], morph):
    lemmatized_text = ""
    for sentence in text:
        tokens = tokenize.word_tokenize(sentence)
        lemmas = ''
        for token in tokens:
            lemma = morph.normal_forms(token)[0]
            if lemma not in stopwords.words('russian'):
                lemmas += lemma + ' '
        lemmas = lemmas.rstrip()
        lemmatized_text += lemmas + " "
    lemmatized_text.rstrip()
    return lemmatized_text