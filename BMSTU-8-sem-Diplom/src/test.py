from matplotlib import pyplot as plt
from classification import get_train_data, svm_classification
from dataset import make_data_test
from pymorphy2 import MorphAnalyzer
import time

morph = MorphAnalyzer()
num = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
def make_data():
    #Make data tests
    # make_data_test(10, morph)
    # make_data_test(20, morph)
    # make_data_test(40, morph)
    # make_data_test(100, morph)
    # make_data_test(200, morph)
    # make_data_test(1000, morph)
    make_data_test(400, morph)
    make_data_test(600, morph)
    make_data_test(800, morph)

def test1():
    train_score = []
    test_score = []
    f1_score = []
    time1 = []
    for count in num:
        start = time.time()
        text_train, text_test, title_train, title_test = get_train_data('data_test/data_' + str(count) + '.csv', morph)
        _, _, train, test, f1 = svm_classification(
                text_train, text_test,
                title_train, title_test
            )
        end = time.time()
        train_score.append(train)  
        test_score.append(test)  
        f1_score.append(f1)  
        time1.append(end - start)
    print(test_score)
    print(f1_score)
    print(time1)

    line1, = plt.plot(num, test_score, label = 'Аккуратность на тестовой выборке')
    line2, = plt.plot(num, f1_score, label = 'F1-мера')
    plt.xlabel('Количество текстов')
    plt.ylabel('Точность')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.show()

    plt.plot(num, time1)
    plt.xlabel('Количество текстов')
    plt.ylabel('Время обучения')
    plt.grid()
    plt.show()

def test2():
    train_score = []
    test_score = []
    f1_score = []
    train_stop_score = []
    test_stop_score = []
    f1_stop_score = []
    for count in num:
        text_train, text_test, title_train, title_test = get_train_data('data_test/data_' + str(count) + '.csv', morph)
        _, _, train, test, f1 = svm_classification(
                text_train, text_test,
                title_train, title_test
            )
        train_score.append(train)  
        test_score.append(test)  
        f1_score.append(f1)  
    for count in num:
        text_train, text_test, title_train, title_test = get_train_data('data_test/data_stop_' + str(count) + '.csv', morph)
        _, _, train, test, f1 = svm_classification(
                text_train, text_test,
                title_train, title_test
            )
        train_stop_score.append(train)  
        test_stop_score.append(test)  
        f1_stop_score.append(f1)  
    print(test_score)
    print(train_stop_score)
    print(f1_score)
    print(f1_stop_score)

    line1, = plt.plot(num, test_score, label = 'Тестовая выборка без стоп-слов')
    line2, = plt.plot(num, test_stop_score, label = 'Тестовая выборка с стоп-словами')
    plt.xlabel('Количество текстов')
    plt.ylabel('Точность')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.show()

    line1, = plt.plot(num, f1_score, label = 'Без стоп-слов')
    line2, = plt.plot(num, f1_stop_score, label = 'С стоп-словами')
    plt.xlabel('Количество текстов')
    plt.ylabel('F1-мера')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.show()

def test3():
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    train_score = [[] for i in range(len(kernel))]
    test_score = [[] for i in range(len(kernel))]
    f1_score = [[] for i in range(len(kernel))]
    
    # print(f1_score)
    for i in range(len(kernel)):
        for count in num:
            text_train, text_test, title_train, title_test = get_train_data('data_test/data_' + str(count) + '.csv', morph)
            _, _, train, test, f1 = svm_classification(
                    text_train, text_test,
                    title_train, title_test,
                    kernel[i]
                )
            train_score[i].append(train)  
            test_score[i].append(test)  
            f1_score[i].append(f1)  
    # print(train_score[0])
    line1, = plt.plot(num, test_score[0], marker = 'o', label = 'Линейное')
    line2, = plt.plot(num, test_score[1], marker = '*', label = 'Полиномиальное')
    line3, = plt.plot(num, test_score[2], marker = 'x', label = 'Гауссово RBF')
    line4, = plt.plot(num, test_score[3], marker = '+', label = 'Сигмоидальное')
    plt.xlabel('Количество текстов')
    plt.ylabel('Точность в тестовой выборке')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()

    line1, = plt.plot(num, f1_score[0], marker = 'o', label = 'Линейное')
    line2, = plt.plot(num, f1_score[1], marker = '*', label = 'Полиномиальное')
    line3, = plt.plot(num, f1_score[2], marker = 'x', label = 'Гауссово RBF')
    line4, = plt.plot(num, f1_score[3], marker = '+', label = 'Сигмоидальное')
    plt.xlabel('Количество текстов')
    plt.ylabel('F1-мера')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()
    
# make_data()
# test1()
# test2()
# test3()

y = [0.15, 0.11, 0.3, 2.02, 6.50, 21.28, 47.39, 82.76, 127.45]
plt.plot(num, y)
plt.xlabel('Количество текстов, штук')
plt.ylabel('Время обучения, c')
plt.grid()
plt.show()

test_score = [0.2, 0.9, 0.9, 0.92, 0.92, 0.9175, 0.94, 0.9425, 0.943]
f1_score = [0.26, 0.89, 0.89, 0.90, 0.91, 0.92, 0.94, 0.942, 0.943]

line1, = plt.plot(num, test_score, marker = 'o', label = 'Аккуратность на тестовой выборке')
line2, = plt.plot(num, f1_score, marker = 'x', label = 'F1-мера')
plt.xlabel('Количество текстов, штук')
plt.ylabel('Точность')
plt.legend(handles=[line1, line2])
plt.grid()
plt.show()