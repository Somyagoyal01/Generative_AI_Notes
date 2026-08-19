import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
st.title("Passenger Survival Chance in the Titanic Journey")
pclass=st.slider('Enter the passenger class for the user',1,3)
sex=st.selectbox('Enter the passenger gender',['male','female'])
sibsp=st.slider('Enter the number of siblings/spouses aboard the Titanic',0,8)
parch=st.slider('Enter the number of parents/children aboard the Titanic',0,6)
fare=st.number_input("Enter the fare paid by passenger")
embarked=st.selectbox('Enter the port of embarkation',['Cherbourg','Queenstown','Southampton'])
data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])
if st.button('Data'):
    st.write(data)
model=load_model('model.h5')
with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)
with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)
data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])
y=model.predict(data)
y=y[0][0]
def Chance(y):
    if y>0.5:
        return "The passenger has a high chance of survival"
    else:
        return "The passenger has a low chance of survival"
if st.button('Predict Survival Chance'):
    st.write('Probability of passenger survival chance:',y)
    st.write(Chance(y))

