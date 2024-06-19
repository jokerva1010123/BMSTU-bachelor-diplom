# Text Classification

'Text Classfication' using SVM and Naive Bayes.

The input file is also uploaded - corpus.csv

# Version 2:

training.py - Thao tác này sẽ huấn luyện một mô hình trên dữ liệu của bạn và lưue the trained LabelEncoder, TFIDF Vector and the model itself

inference.py - Thao tác này sẽ tải các tệp đã lưu này vào đĩa và đưa ra dự đoán.

# Text-Classification-SVM

# Text-Classification-SVM

python3 -m spacy download en_core_web_lg

SELECT title, category
FROM dsjvoxarticles
WHERE category LIKE '%business%'
OR category LIKE '%entertainment%'
OR category LIKE '%politics%'
OR category LIKE '%sport%'
OR category LIKE '%tech%' LIMIT 1000000;
