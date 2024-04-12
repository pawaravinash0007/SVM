import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("diabetic.pkl","rb"))

def predict(data):
    clf = pickle.load(open("diabetic.pkl","rb"))
    return clf.predict(data)


st.title("Identify Diabetic Peoples using Machine Learning")
st.markdown("This Model Identify weather person suffering from Diabetic or not")

st.header("Medical Parameters")
col1,col2 = st.columns(2)

with col1:
	st.text("Pregnancies")
	el=st.number_input("Pregnancies", 1,24,2)
	st.text("Glucose")
	tos = st.slider("Glucose", 1.0, 6.0, 0.5)
	st.text("BloodPressure")
	top = st.slider("BloodPressure", 0.0, 12.0, 0.5)
	st.text("DiabetesPedigreeFunctionDiabetesPedigreeFunction")
	top1 = st.slider("DiabetesPedigreeFunction", 0.0, 2.329, 0.5)
	
with col2:
	st.text("Age")
	ag=st.number_input("Age",0,68,2)
	st.text("SkinThickness")
	gr = st.slider("SkinThickness", 1.0,47.0,0.5)
	st.text("Insulin")
	gr1 = st.slider("Insulin", 1.0,16.0,0.5)
	st.text("BMI")
	bm1 = st.slider("BMI", 1.0,57.3,0.5)

st.text('')
if st.button("Predict Performance Rate"):
    result = clf.predict(
        np.array([[el,tos,top,top1,ag,gr,gr1,bm1]]))
    st.text(result[0])
st.markdown("Developed by External Guide Avinash Pawar and WBL Intern at NIELIT Daman")
