import docx2txt
from flask import * 
import os
from flask import render_template, request, redirect, url_for,Flask, request, jsonify, render_template

import numpy as np #imported numpy and Assigned as np
import pandas as pd #imported pandas and Assigned as pd
import pickle

app = Flask(__name__)

##App Imported Pickles In App
print("App Imported Pickles In App")
multi = pickle.load(open('multinb.pkl', 'rb'))
linear=pickle.load(open('linearsvc.pkl', 'rb'))
random = pickle.load(open('random.pkl', 'rb'))
logic = pickle.load(open('logic.pkl', 'rb'))

##Other Pickle Objects are imported
id_to_category=pickle.load(open('id_to_category.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))
tfidf=pickle.load(open('tfidf.pkl', 'rb'))


#@app.route('/')
#def home():
 #   return render_template('index.html')
    
@app.route('/',methods=['POST','GET'])
def loginacc():
    return render_template('login.html')  
    
    
@app.route('/app',methods=['POST','GET'])
def home():
    return render_template('index.html')   
    
    


@app.route('/recheck',methods=['POST','GET'])
def recheck():
    return render_template('index.html')


#@app.route('/loginacc',methods=['POST','GET'])
#def loginacc():
 #   return render_template('login.html')

@app.route('/filedata',methods=['POST','GET'])
def filedata():
    return render_template('files.html')
@app.route('/cvr.ac.in',methods=['POST','GET'])
def cvrweb():
    return redirect(url_for('http://cvr.ac.in/home4/'))

@app.route('/code_login')
def codelogin():
    return render_template('code_login.html')


@app.route('/Get_code')
def getcode():
    return render_template('github.com/pskr0/MajorProject.git')



@app.route('/predict',methods=['POST','GET'])
def predicts():  
    if request.method == 'POST':  
        f = request.files['file']  
        a=f.save(f.filename)
        filenme=f.filename
    file_text = docx2txt.process(filenme)
    
    file_text=str(file_text)
    
    print("Multinomial NB Prediction")
    # Multinomial NB Prediction
    multi_file_input=(multi.predict(vectorizer.transform([file_text])))
    
    print("RANDOMFOREST Prediction")      
    #RANDOMFOREST
    random_out=(random.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, random_out):
      random_file_input=id_to_category[predicted]
      
    print("LogisticRegression Prediction")    
    
    #LogisticRegression
    logic_out=(logic.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, logic_out):
      logic_file_input=id_to_category[predicted]      
      
    print("LinearSVC Prediction")       
    #LinearSVC
    linear_out=(linear.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, linear_out):
      linear_file_input=id_to_category[predicted]
      
    text_all_file_inputs=[multi_file_input,random_file_input,logic_file_input,linear_file_input]
    
    multi_count_file = text_all_file_inputs.count(multi_file_input)
    
    random_count_file = text_all_file_inputs.count(random_file_input)
    
    logic_count_file = text_all_file_inputs.count(logic_file_input)
   
    linear_count_file = text_all_file_inputs.count(linear_file_input)
    

    a2=(multi_count_file/4)*100

    b2=(random_count_file/4)*100

    c2=(logic_count_file/4)*100

    d2=(linear_count_file/4)*100
    
     
    if a2 >=75:
      Finalresult_file=multi_file_input
      
    if b2 >=75:
      Finalresult_file=random_file_input
      
    if c2>=75:
      Finalresult_file=logic_file_input      
      
      
    if a2 ==50:
      Finalresult_file=multi_file_input
    if b2 ==50:
      Finalresult_file=random_file_input
    if c2==50:
      Finalresult_file=logic_file_input
    if d2==50:
      Finalresult_file=linear_file_input
      
    if a2==25 and b2==25 and c2== 25 and d2==25:
      Finalresult_file=linear_file_input
    os.remove(filenme)    
    
    return render_template('conform.html',Final_file='{} '.format(Finalresult_file),filenme='{} '.format(filenme),multi_file_input='{}'.format(multi_file_input),random_file_input='{}'.format(random_file_input),logic_file_input='{}'.format(logic_file_input),linear_file_input='{}'.format(linear_file_input),a2='{} '.format(a2),b2='{} '.format(b2),c2='{} '.format(c2),d2='{} '.format(d2))
    
    
if __name__ == "__main__":
     app.run(host="127.0.0.1",port=6442)
