from flask import Flask, render_template, url_for, request, send_file, url_for
from werkzeug.utils import secure_filename
import svm.inference as classifier
from flask_cors import CORS, cross_origin
# import those packages you have already imported

app = Flask(__name__)
CORS(app, support_credentials=True)
HTTP_METHODS = ['GET', 'HEAD', 'POST', 'PUT',
                'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH']


@app.route('/classifier')
def index():
    return render_template("classifier.html")


@app.route('/classifier')
def classify():
    return render_template("classifier.html")


@app.route('/classifier', methods=['POST'])
def classify2():
    if request.method == 'POST':
        raw_text = request.form['rawtext']
        check_text = classifier.textClassifier(raw_text)
        my_list = ['business', 'entertainment', 'politics', 'sport', 'tech']
        check = any(element in check_text for element in my_list)
        if check:
            result = check_text
        else:
            result = "No classifier"
        return render_template("classifier.html", results=result, raw_text=raw_text)


# http://127.0.0.1:8088/classifier
if __name__ == '__main__':
    app.run(debug=False, threaded=False, port=8088)
