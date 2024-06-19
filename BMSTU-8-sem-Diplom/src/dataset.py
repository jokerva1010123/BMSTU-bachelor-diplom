import pandas as pd
import random
from preprocessing import split_text, lemmatize_remove_stopwords, lemmatize
import time
from pymorphy2 import MorphAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

def check():
    Corpus = pd.read_csv("./data/sport2.csv", low_memory=False)
    print(Corpus.isnull().sum())
    Corpus = pd.read_csv("./data/culture2.csv", low_memory=False)
    print(Corpus.isnull().sum())
    Corpus = pd.read_csv("./data/world2.csv", low_memory=False)
    print(Corpus.isnull().sum())
    Corpus = pd.read_csv("./data/tech2.csv", low_memory=False)
    print(Corpus.isnull().sum())
    Corpus = pd.read_csv("./data/economy2.csv", low_memory=False)
    print(Corpus.isnull().sum())
    Corpus = pd.read_csv("./data/dataset.csv", low_memory=False)
    print(Corpus.isnull().sum())

def abc(file):
    Corpus = pd.read_csv(file, low_memory=False)

    columns = ['text', 'topic']

    sport = Corpus[Corpus["topic"] == 'Спорт']
    economy = Corpus[Corpus["topic"] == 'Экономика']
    culture = Corpus[Corpus["topic"] == 'Культура']
    tech = Corpus[Corpus["topic"] == 'Наука и техника']
    world = Corpus[Corpus["topic"] == 'Мир']

    # sport.to_csv("data/sport.csv", index=False, columns=columns)
    # economy.to_csv("data/economy.csv", index=False, columns=columns)
    # culture.to_csv("data/culture.csv", index=False, columns=columns)
    # tech.to_csv("data/tech.csv", index=False, columns=columns)
    # world.to_csv("data/world.csv", index=False, columns=columns)
    
    file_names = ['data/sport.csv', 'data/economy.csv', 'data/culture.csv', 'data/tech.csv', 'data/world.csv']
    dataframes = []
    for file in file_names:
        df = pd.read_csv(file)
        dataframes.append(df)
    merged_df = pd.concat(dataframes)
    merged_df.to_csv('data/dataset.csv', index=False)

def make_dataset(num, morph):
    Corpus = pd.read_csv("./data/dataset.csv", low_memory=False)

    sport = Corpus[Corpus["topic"] == 'Спорт']
    economy = Corpus[Corpus["topic"] == 'Экономика']
    culture = Corpus[Corpus["topic"] == 'Культура']
    tech = Corpus[Corpus["topic"] == 'Наука и техника']
    world = Corpus[Corpus["topic"] == 'Мир']

    random_sport = random.sample(range(len(sport)), num)
    random_economy = random.sample(range(len(economy)), num)
    random_culture = random.sample(range(len(culture)), num)
    random_tech = random.sample(range(len(tech)), num)
    random_world = random.sample(range(len(world)), num)

    sport = sport.iloc[random_sport]
    economy = economy.iloc[random_economy]
    culture = culture.iloc[random_culture]
    tech = tech.iloc[random_tech]
    world = world.iloc[random_world]
    all = [sport, economy, culture, tech, world]

    # columns = ['text', 'topic']
    # sport.to_csv("data/sport2.csv", index=False, columns=columns)
    # economy.to_csv("data/economy2.csv", index=False, columns=columns)
    # culture.to_csv("data/culture2.csv", index=False, columns=columns)
    # tech.to_csv("data/tech2.csv", index=False, columns=columns)
    # world.to_csv("data/world2.csv", index=False, columns=columns)

    start = time.time()
    dataframe = pd.DataFrame()
    title = ['Спорт'] * num + ['Экономика'] * num + ['Культура'] * num + ['Наука и техника'] * num + ['Мир'] * num
    texts = []
    for type in all:
        all_text = type['text']
        for text in all_text:
            split = split_text(text)
            lemma = lemmatize_remove_stopwords(split, morph)
            texts.append(lemma)

    dataframe['text'] = texts
    dataframe['topic'] = title
    end = time.time()
    print(end - start)

    name = "data/data_" + str(num*5) + '.csv'
    dataframe.to_csv(name, index=False)

def make_5_topic_all_dataset():
    Corpus = pd.read_csv("./data/dataset.csv", low_memory=False)

    topics = ['Спорт', 'Экономика', 'Культура', 'Наука и техника', 'Мир']
    column = ['text', 'topic']

    df = Corpus[Corpus['topic'].isin(topics)]
    df = df[column]

    df.to_csv('data/dataset.csv', index=False)

def make_data_test(num, morph):
    Corpus = pd.read_csv("./data/dataset.csv", low_memory=False)

    sport = Corpus[Corpus["topic"] == 'Спорт']
    economy = Corpus[Corpus["topic"] == 'Экономика']
    culture = Corpus[Corpus["topic"] == 'Культура']
    tech = Corpus[Corpus["topic"] == 'Наука и техника']
    world = Corpus[Corpus["topic"] == 'Мир']

    random_sport = random.sample(range(len(sport)), num)
    random_economy = random.sample(range(len(economy)), num)
    random_culture = random.sample(range(len(culture)), num)
    random_tech = random.sample(range(len(tech)), num)
    random_world = random.sample(range(len(world)), num)

    sport = sport.iloc[random_sport]
    economy = economy.iloc[random_economy]
    culture = culture.iloc[random_culture]
    tech = tech.iloc[random_tech]
    world = world.iloc[random_world]
    all = [sport, economy, culture, tech, world]

    start = time.time()
    dataframe1 = pd.DataFrame()
    dataframe2 = pd.DataFrame()
    title = ['Спорт'] * num + ['Экономика'] * num + ['Культура'] * num + ['Наука и техника'] * num + ['Мир'] * num
    texts1 = []
    texts2 = []
    for type in all:
        all_text = type['text']
        for text in all_text:
            split = split_text(text)
            lemma1 = lemmatize_remove_stopwords(split, morph)
            lemma2 = lemmatize(split, morph)
            texts1.append(lemma1)
            texts2.append(lemma2)

    dataframe1['text'] = texts1
    dataframe1['topic'] = title
    dataframe2['text'] = texts2
    dataframe2['topic'] = title
    end = time.time()
    # print(end - start)
    dataframe1.to_csv('data_test/data_' + str(num * 5) + '.csv', index=False)
    dataframe2.to_csv('data_test/data_stop_' + str(num * 5) + '.csv', index=False)

def diagram():
    Corpus = pd.read_csv("./data/dataset.csv", low_memory=False)

    sport = Corpus[Corpus["topic"] == 'Спорт']
    economy = Corpus[Corpus["topic"] == 'Экономика']
    culture = Corpus[Corpus["topic"] == 'Культура']
    tech = Corpus[Corpus["topic"] == 'Наука и техника']
    world = Corpus[Corpus["topic"] == 'Мир']

    print(len(tech))
    print(len(sport))
    print(len(economy))
    print(len(culture))
    print(len(world))

if __name__ == '__main__':
    # morph = MorphAnalyzer()
    # make_5_topic_all_dataset()
    # make_dataset(100, morph)
    # make_dataset(200, morph)
    # make_dataset(500, morph)
    # make_dataset(1000, morph)
    diagram()