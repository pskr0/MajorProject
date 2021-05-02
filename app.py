import numpy as np
import datetime
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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
    prediction=(clf.predict(count_vect.transform([a])))
    #prediction = model.predict(a)
    return render_template('conform.html',dat2="{} ".format(prediction))

if __name__ == "__main__":
     #app.run(debug=True)
     app.run(host="0.0.0.0",port=6442)
