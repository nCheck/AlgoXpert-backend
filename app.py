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
from preprocessor import preprocess
from classifier import classify



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
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'File successfully uploaded' , 'filename' : filename})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp




@app.route('/predict', methods=['GET'])
def predict():

    # print( json.dumps( request.json['data'] ) )

    try :
        print("hi")

        LEARNING = request.args.get('learning')
        TARGET = request.args.get('target')
        UNWANTED = []
        print(LEARNING , TARGET)
        if LEARNING == 'supervised':

            x_train,x_test,y_train,y_test = preprocess('data.txt', TARGET , UNWANTED )

            # print(x_train)

            result = classify( x_train,x_test,y_train,y_test )
            labels = list(result.keys())
            values = list(result.values())

            for i in range(4):
                values[i] = 100 * values[i]

            print(labels)
            print(values)
            return render_template('index.html' , labels = labels , values = values)


        else:

            data = preprocess('data.txt', None , UNWANTED )

            return jsonify( { "result" : "Under Construction" } )

    except Exception as e:
        return jsonify( { "result" : "error" , "status"  : False  } )      





@app.route('/', methods=['GET'])
def home():
    print("loaded")
    return render_template('index.html')




if __name__ == '__main__':
    app.run()