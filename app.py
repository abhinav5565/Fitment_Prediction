from ast import Try
import os
from telnetlib import X3PAD
import warnings
from flask import Flask, render_template, url_for, request, redirect,session
import pandas as pd
from gtts import gTTS
import joblib
import playsound
g=joblib.load('fitment_pred.pkl')
l1=joblib.load('output1.pkl')
l2=joblib.load('output2.pkl')
l3=joblib.load('output3.pkl')
l4=joblib.load('output4.pkl')
l5=joblib.load('output5.pkl')
l6=joblib.load('output6.pkl')
l7=joblib.load('output7.pkl')
l8=joblib.load('output8.pkl')
l9=joblib.load('output9.pkl')

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
# secret key is needed for session
@app.route('/',  methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route('/prediction')
def prediction():
    try:
        name = session['name']
        language = session['language']
        age = int(session['age'])
        sex = session['sex']
        hd = session['hd']
        field = session['field']
        institute = session['institute']
        cgpa = int(session['cgpa'])
        current_ctc = int(session['current_ctc'])
        ctc = int(session['ctc'])
        role_factor = session['role_factor']
        experience = int(session['experience'])
        marital_status = session['marital_status']
        emp_score = int(session['emp_score'])
        current_designation = session['current_designation']
        current_company = session['current_company']
        department_in_company=session['department_in_company']
        total_leaves_taken=int(session['total_leaves_taken'])

        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
        a8=[]
        a9=[]
        a0=[]
        a11=[]
        a12=[]
        a13=[]
        a14=[]
        a15=[]
        a16=[]
        a17=[]
        a18=[]
        a19=[]
        a20=[]
        a21=[]
        a22=[]
        a23=[]
        a24=[]
        a25=[]
        a26=[]
        a27=[]
        a28=[]
        a1.append(name)
        a2.append(language)
        a3.append(age)
        a4.append(sex)
        a5.append(hd)
        a6.append(field)
        a7.append(institute)
        a8.append(cgpa)
        a9.append(current_ctc)
        a0.append(ctc)
        if(role_factor=='CurrentCompanyType'): 
            a11.append(1) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='DegreeBranch'):
            a11.append(0) 
            a12.append(1)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='EmpScore'):
            a11.append(0) 
            a12.append(0)
            a13.append(1)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='Ethnicity'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(1)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='Gender'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(1)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='HighestDegree'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(1)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='LatestCgpa'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(1)
            a18.append(0)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='MaritalStatus'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(1)
            a19.append(0)
            a20.append(0)
        elif(role_factor=='Nothing'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(1)
            a20.append(0)
        elif(role_factor=='Experience'):
            a11.append(0) 
            a12.append(0)
            a13.append(0)
            a14.append(0)
            a15.append(0)
            a16.append(0)
            a17.append(0)
            a18.append(0)
            a19.append(0)
            a20.append(1)


        a21.append(experience)
        a22.append(marital_status)
        a23.append(emp_score)
        a24.append(current_designation)
        a25.append(current_company)
        a26.append(department_in_company)
        a27.append(total_leaves_taken)
    
        df3=pd.DataFrame({'name':a1,'language':a2,'age':a3,'sex':a4,'hd':a5,'field':a6,'institute':a7,'cgpa':a8,'current_ctc':a9,'ctc':a0,'role_factor1':a11,'role_factor2':a12,'role_factor3':a13,'role_factor14':a14,'role_factor15':a15,'role_factor16':a16,'role_factor17':a17,'role_factor18':a18,'role_factor19':a19,'role_factor20':a20,'experience':a21,'marital_status':a22,'emp_score':a23,'current_designation':a24,'current_company':a25,'department_in_company':a26,'total_leaves_taken':a27})
        

        
        x2=df3['language']
        x3=df3['age']
        x4=df3['sex']
        x5=df3['hd']
        x6=df3['field']
        x7=df3['institute']
        x8=df3['cgpa']
        x9=df3['current_ctc']
        x0=df3['ctc']
        x11=df3['role_factor1']
        x12=df3['role_factor2']
        x13=df3['role_factor3']
        x14=df3['role_factor14']
        x15=df3['role_factor15']
        x16=df3['role_factor16']
        x17=df3['role_factor17']
        x18=df3['role_factor18']
        x19=df3['role_factor19']
        x20=df3['role_factor20']
        x21=df3['experience']
        x22=df3['marital_status']
        x23=df3['emp_score']
        x24=df3['current_designation']
        x25=df3['current_company']
        x26=df3['department_in_company']
        x27=df3['total_leaves_taken']
    
    
    
        
        x2=l1.transform(x2)
    # x3=t.transform(x3)
        x4=l2.transform(x4)
        x5=l3.transform(x5)
        x6=l4.transform(x6)
        x7=l5.transform(x7)
        x22=l6.transform(x22)
        x24=l7.transform(x24)
        x25=l8.transform(x25)
        x26=l9.transform(x26)
        #x8=t.transform(x8)
    # x9=t.transform(x9)
    #  x0=t.transform(x0)

        
        df3['language']=x2
        df3['age']=x3
        df3['sex']=x4
        df3['hd']=x5
        df3['field']=x6
        df3['institute']=x7
        df3['cgpa']=x8
        df3['current_ctc']=x9
        df3['ctc']=x0
        df3['role_factor1']=x11
        df3['role_factor2']=x12
        df3['role_factor3']=x13
        df3['role_factor14']=x14
        df3['role_factor15']=x15
        df3['role_factor16']=x16
        df3['role_factor17']=x17
        df3['role_factor18']=x18
        df3['role_factor19']=x19
        df3['role_factor20']=x20
        df3['experience']=x21
        df3['marital_status']=x22
        df3['emp_score']=x23
        df3['current_designation']=x24
        df3['current_company']=x25
        df3['department_in_company']=x26
        df3['total_leaves_taken']=x27

        x=df3.drop(['name'],axis=1)


        y8=g.predict(x)
        k=y8[0]
        m=k
        display="Thankyou. Here is the information you needed."
        f="Fitmennt Prediction {} ".format(m)
        tts=gTTS(f,lang='en')
        files="summarys.mp3"
        tts.save(files)
        playsound.playsound(files)
        os.remove(files)
        return render_template('prediction.html',category=f,display=display)
    except:
        display="Either invalid data or could not get the required information"
        n=""
        f=""
        return render_template('prediction.html',display=display,category=f)
    
   
    

@app.route('/book',  methods=['POST'])
def bookgenre():
    session['name']=request.form.get('name')
    session['language']=request.form.get('language')
    session['age']=request.form.get('age')
    session['sex']=request.form.get('sex')
    session['hd']=request.form.get('hd')
    session['field']=request.form.get('field')
    session['institute']=request.form.get('institute')
    session['cgpa']=request.form.get('cgpa')
    session['current_ctc']=request.form.get('current_ctc')
    session['ctc']=request.form.get('ctc')
    session['role_factor']=request.form.get('role_factor')
    session['experience']=request.form.get('experience')
    session['marital_status']=request.form.get('marital_status')
    session['emp_score']=request.form.get('emp_score')
    session['current_designation']=request.form.get('current_designation')
    session['current_company']=request.form.get('current_company')
    session['department_in_company']=request.form.get('department_in_company')
    session['total_leaves_taken']=request.form.get('total_leaves_taken')
    return redirect(url_for("prediction"))
    

