from flask import Flask,render_template,request
import pickle
import numpy as np
import bz2file as bz2

app=Flask(__name__)

def decompress_pickle(file):
    data=bz2.BZ2File(file,'rb')
    data=pickle.load(data)
    return data
model=decompress_pickle('creditscore.pkl.pbz2')

@app.route('/')
def hello():
    return render_template('creditscore.html')	
@app.route('/predict',methods=['POST'])
def predict():
   
   Annual_Income=float(request.form['Annual_Income'])
   Num_Bank_Accounts=float(request.form['Num_Bank_Accounts'])
   Num_Credit_Card =float(request.form['Num_Credit_Card'])
   Interest_Rate =float(request.form['Interest_Rate'])
   Num_of_Loan =float(request.form['Num_of_Loan'])
   Delay_from_due_date =float(request.form['Delay_from_due_date'])
   Num_of_Delayed_Payment =float(request.form['Num_of_Delayed_Payment'])
   Changed_Credit_Limit =float(request.form['Changed_Credit_Limit'])
   Num_Credit_Inquiries  =float(request.form['Num_Credit_Inquiries'])
   Credit_Mix =request.form['Credit_Mix']
   if Credit_Mix =='Bad':
     Credit_Mix =0
   elif Credit_Mix =='Good':
     Credit_Mix =1
   elif Credit_Mix =='Standard':
     Credit_Mix =2
   Outstanding_Debt =float(request.form['Outstanding_Debt'])
   Total_EMI_per_month =float(request.form['Total_EMI_per_month'])

   print(Annual_Income,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Changed_Credit_Limit,Num_Credit_Inquiries,Credit_Mix,Outstanding_Debt,Total_EMI_per_month)

   

   feature=np.array([[Annual_Income,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Changed_Credit_Limit,Num_Credit_Inquiries,Credit_Mix,Outstanding_Debt,Total_EMI_per_month]])

   scores=model.predict(feature)						

    				

   return render_template('creditscore.html',predicted=scores)	

if __name__=='__main__':						
    app.run(debug=True)