import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
st.title("CREDIT CARD FRAUD DETECTION")
data=pd.read_excel(r"new.xlsx")
# print(data)
X=data[['Amount','TransactionType', 'Location']]
y=data['IsFraud']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=LogisticRegression()
model.fit(X_train,y_train)

test_pred=model.predict(X_test)

train_pred=model.predict(X_train)

train_report=classification_report(y_train,train_pred)

test_report=classification_report(y_test,test_pred)

st.write(train_report)
st.write(test_report)

amount=st.number_input("Enter your amount   :")
t_type=st.number_input("Enter your trancsaction type :")
location=st.number_input("Enter your location    :")
bt=st.button("Submit")
if bt:
    output=model.predict([[amount,t_type,location]])
    st.write(output)
