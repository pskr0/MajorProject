import docx2txt
from flask import * 
import os
from flask import render_template, request, redirect, url_for


# replace following line with location of your .docx file
#file_text = docx2txt.process(b)


import numpy as np #imported numpy and Assigned as np
import pandas as pd #imported pandas and Assigned as pd
import pickle
from flask import Flask, request, jsonify, render_template

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
    
@app.route('/finals',methods=['POST','GET'])
def final(Final_text):
    return render_template('final.html',Final_text='{}%'.format(Finalresult_text),Final_file='{}%'.format(Finalresult_file),filenme='{}%'.format(filenme))
    #return render_template('final.html')




@app.route('/predict',methods=['POST','GET'])
def predicts():  
    if request.method == 'POST':  
        f = request.files['file']  
        a=f.save(f.filename)
        filenme=f.filename
    file_text = docx2txt.process(filenme)
    
    text_input = request.form['inputdata'] 
    text_input=str(text_input)
    file_text=str(file_text)
    
    print("Multinomial NB Prediction")
    # Multinomial NB Prediction
    multi_text_input=(multi.predict(vectorizer.transform([text_input])))
    multi_file_input=(multi.predict(vectorizer.transform([file_text])))
    
    print("RANDOMFOREST Prediction")    
    #RANDOMFOREST MODEL Prediction
    random_out=(random.predict(tfidf.transform([text_input])))
    for text, predicted in zip(text_input, random_out):
      random_text_input=id_to_category[predicted]
    
    #RANDOMFOREST
    ramdom_out=(random.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, random_out):
      random_file_input=id_to_category[predicted]
      
    print("LogisticRegression Prediction")    
    #LogisticRegression MODEL Prediction
    logic_out=(logic.predict(tfidf.transform([text_input])))
    for text, predicted in zip(text_input, logic_out):
      logic_text_input=id_to_category[predicted]
    
    #LogisticRegression
    logic_out=(logic.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, logic_out):
      logic_file_input=id_to_category[predicted]      
      
    print("LinearSVC Prediction")    
    #LinearSVC MODEL Prediction
    linear_out=(linear.predict(tfidf.transform([text_input])))
    for text, predicted in zip(text_input, linear_out):
      linear_text_input=id_to_category[predicted]
    
    #LinearSVC
    linear_out=(linear.predict(tfidf.transform([file_text])))
    for text, predicted in zip(file_text, linear_out):
      linear_file_input=id_to_category[predicted]
      
    #Finding Count Of Each Models
    text_all_text_inputs=[multi_text_input,random_text_input,logic_text_input,linear_text_input]
    text_all_file_inputs=[multi_file_input,random_file_input,logic_file_input,linear_file_input]
    
    multi_count_text = text_all_text_inputs.count(multi_text_input)
    multi_count_file = text_all_file_inputs.count(multi_file_input)
    
    random_count_text = text_all_text_inputs.count(random_text_input)
    random_count_file = text_all_file_inputs.count(random_file_input)
    
    logic_count_text = text_all_text_inputs.count(logic_text_input)
    logic_count_file = text_all_file_inputs.count(logic_file_input)
    
    linear_count_text = text_all_text_inputs.count(linear_text_input)
    linear_count_file = text_all_file_inputs.count(linear_file_input)
    
    a1=(multi_count_text/4)*100
    a2=(multi_count_file/4)*100
    
    b1=(random_count_text/4)*100
    b2=(random_count_file/4)*100

    c1=(logic_count_text/4)*100
    c2=(logic_count_file/4)*100

    d1=(linear_count_text/4)*100
    d2=(linear_count_file/4)*100
    
    if a1 >=75:
      Finalresult_text=random_text_input
    if a1 ==50:
      Finalresult_text=random_text_input
    if b1 ==50:
      Finalresult_text=linear_text_input
    if c1==50:
      Finalresult_text=logic_text_input
    if d1==50:
      Finalresult_text=linear_count_text
    if a1==25 and b1==25 and c1 == 25 and d1==25:
      Finalresult_text=random_text_input
     
    if a2 >=75:
      Finalresult_file=random_file_input
    if a2 ==50:
      Finalresult_file=random_file_input
    if b2 ==50:
      Finalresult_file=linear_file_input
    if c2==50:
      Finalresult_file=logic_file_input
    if d2==50:
      Finalresult_file=linear_count_text
    if a2==25 and b2==25 and c2== 25 and d2==25:
      Finalresult_file=random_file_input
        

     
    
    return render_template('conform.html',filenme="{} ".format(filenme),multi_file_input='{}'.format(multi_file_input),multi_text_input='{}'.format(multi_text_input),random_file_input='{}'.format(random_file_input),random_text_input='{}'.format(random_text_input),logic_file_input='{}'.format(logic_file_input),logic_text_input='{}'.format(logic_text_input),linear_file_input='{}'.format(linear_file_input),linear_text_input='{}'.format(linear_text_input),a1='{} '.format(a1),a2='{} '.format(a2),b1='{} '.format(b1),b2='{} '.format(b2),c1='{} '.format(c1),c2='{} '.format(c2),d1='{} '.format(d1),d2='{} '.format(d2))
    return redirect(url_for('final',Final_text = Finalresult_text))




if __name__ == "__main__":
     #app.run(debug=True)
     app.run(host="127.0.0.1",port=6442)
