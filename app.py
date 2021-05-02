
import numpy as np #imported numpy and Assigned as np
import pandas as pd #imported pandas and Assigned as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recheck',methods=['POST','GET'])
def recheck():
    return render_template('index.html')


@app.route('/loginacc',methods=['POST','GET'])
def loginacc():
    return render_template('login.html')

@app.route('/filedata',methods=['POST','GET'])
def filedata():
    return render_template('files.html')
@app.route('/cvr.ac.in',methods=['POST','GET'])
def cvrweb():
    return redirect(url_for('http://cvr.ac.in/home4/'))

@app.route('/code_login')
def codelogin():
    return render_template('code_login.html')



@app.route('/predict',methods=['POST','GET'])
def predicts():  
    projectpath = request.form['inputdata'] 
    a=str(projectpath)
    prediction=(model.predict(vectorizer.transform([a])))
    #prediction = model.predict[a]
    return render_template('conform.html',dat2="{} ".format(prediction))

if __name__ == "__main__":
     #app.run(debug=True)
     app.run(host="127.0.0.1",port=6442)
