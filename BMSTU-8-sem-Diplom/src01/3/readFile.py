
from collections import defaultdict
from nltk.corpus import wordnet as wn
import json
import re

# WordNetLemmatizer : Xác định danh , tính ,động từ
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Chuyển toàn bộ văn bản sang chữ thường
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(",", " ")

    return str(text)

# Inference


filepath = "C:/Text_Class_SVM/Text-Classification-SVM/News_Category_Dataset_v3.txt"
with open(filepath) as fp:
    line = fp.readline()
    cnt = 1
    while line:
        parsed_data = json.loads(line)
        data = "99999," + text_preprocessing(parsed_data['headline']) + " " + \
            text_preprocessing(parsed_data['short_description']) + \
            "," + text_preprocessing(parsed_data['category'])
        file_path = "/Users/dangnguyen/Documents/Project/Text-Classification/dataPlus.txt"
        print("Line {}: {}".format(cnt, text_preprocessing(data)))
        # Open the file in write mode ('w')
        with open(file_path, 'a') as file:
            # Write the data to the file
            file.write(data)
            file.write("\n")
            file.close()
        line = fp.readline()
        cnt += 1
