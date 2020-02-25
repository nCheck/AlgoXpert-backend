import flask
import os
from flask import jsonify, request , render_template
from flask import flash, redirect, url_for, session
from joblib import load
from flask_cors import CORS, cross_origin
import requests, json
import pandas as pd
import requests
import random
from preprocessor import preprocess , findHeaderAndSEP , xnormalize
from classifier import classify
from regressor import regression
from clusterer import clustering
from algo_data import getStats

import re


app = flask.Flask(__name__ , 
            static_url_path='', 
            static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config["DEBUG"] = True
app.secret_key = 'super secret key'
cors = CORS(app, resources={r"/*": {"origins": "*"}})







@app.route('/test', methods=['GET','POST'])
def test():
    data = [ 1 , 2 , "Buckle My Shoe" , 3 , 4 , "Shut the Door" ]
    return jsonify( data )



ALLOWED_EXTENSIONS = set(['csv', 'txt', 'tsv', 'xlsx'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    return 'data.txt'




@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        resp = jsonify({'message' : "WRONG FORM"})
        resp.status_code = 400
        return resp

    myfile = request.files['file']

    if myfile.filename == '' or not allowed_file(myfile.filename):
        resp = jsonify({'message' : "WRONG FORM"})
        resp.status_code = 400
        return resp
    else:

        filename = secure_filename(myfile.filename)
        myfile.save( os.path.join(app.config['UPLOAD_FOLDER'], filename) )
        resp = jsonify({'message' : 'File successfully uploaded' , 'filename' : filename})
        print("success" , resp)
        return redirect(url_for('details'))


@app.route('/details', methods=['GET'])
def details():

    fileloc = 'uploads/data.txt'
    f = open(fileloc , 'r+' , encoding='utf-8-sig')
    line = f.readline().strip()
    f.close()
    f = open(fileloc , 'r+')
    HEADER , SEP = findHeaderAndSEP(f)
    arr = []
    if SEP == '\s+':
        arr = re.split(SEP , line)
    else:
        arr = line.split(SEP)

    cols = pd.read_csv(fileloc , sep=SEP , header=HEADER).columns

    if HEADER is None:
        cols = [ i for i in range( len(arr) ) ]

    f.close()
    print(cols , HEADER , SEP , line)

    return render_template('form.html' , l = len(cols) , cols = cols)



@app.route('/predict', methods=['GET' , 'POST'])
def predict():

    if request.method == 'POST':

        antype = request.form['antype']
        TARGET = request.form['target']

        vals = request.form.keys()
        UNWANTED = []

        try:
            int(TARGET)
            TARGET = int(TARGET)
        except:
            pass

        for v in vals:
            if request.form[v] == 'on':
                UNWANTED.append(v)

        if antype == 'Clustering':

            data = preprocess('data.txt', None , UNWANTED )
            X_principal = xnormalize(data)

            result = clustering(X_principal)
            print(result)
            return render_template('cluster_analysis.html' , result = result)
            # return "Done"

        if antype == 'Association':

            data = preprocess('data.txt', None , UNWANTED )

            return "Under Construction"        

        x_train,x_test,y_train,y_test = preprocess('data.txt', TARGET , UNWANTED )

        # print(y_test.dtype, "y_test")

        if y_test.dtype == 'float64':
            antype = 'Regression'

        if antype == 'Classification':

            result = classify( x_train,x_test,y_train,y_test )
            labels = list(result.keys())
            values = list(result.values())

            for i in range(4):
                values[i] = 100 * values[i]

            print(labels)
            print(values)

            analysis = getStats(labels)

            return render_template('classify_analysis.html' , labels = labels , values = values , analysis = analysis)

        elif antype == 'Regression':

            result = regression( x_train,x_test,y_train,y_test )
            labels = list(result.keys())
            values = list(result.values())

            MAX = max(values) + 4

            print(labels)
            print(values)

            analysis = getStats(labels)

            return render_template('regression_analysis.html' , labels = labels , values = values , analysis = analysis , MAX = MAX )
        
        else:

            return "Invalid Choice"
        
      





@app.route('/', methods=['GET'])
def home():
    print("loaded")
    return render_template('index.html')




if __name__ == '__main__':
    app.run()