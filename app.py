import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
LabelEncoder_category=LabelEncoder()
sc=StandardScaler()


model=pickle.load(open("gender_prediction_model.pkl","rb"))

def show_predict_page():
    st.title("Gender Classification model")
    
    numclaims=st.selectbox("select number of claims",["0","1","2","3","4"])

    veh_body=st.selectbox("select veh_body",["select veh_body",'SEDAN', 'HBACK', 'STNWG', 'TRUCK', 
                           'HDTOP', 'UTE', 'COUPE', 'BUS', 'PANVN', 'MIBUS', 'RDSTR', 'CONVT', 'MCARA'])
    
    veh_age=st.selectbox("select veh_age",["1","2","3","4"])

    area=st.selectbox("select area",['C', 'B', 'F', 'A', 'D', 'E'])

    agecat=st.selectbox("select age_category",["1","2","3","4","5","6"])

    veh_value=st.number_input("veh_values",0.0,34.560000)

    exposure=st.number_input("exposure",0.0,0.9999)

    claimcst0=st.number_input("claims_cost",0.0,55922.129883)

    frequincy=st.number_input("frequincy",0.0,365.250000)

    severity=st.number_input("severity",0.0,55922.0)

    columns=['numclaims','veh_body','veh_age','area','agecat','veh_value','exposure','claimcst0','frequincy','severity']

    ok=st.button("predict Gender")
    if ok:
        row=np.array([numclaims,veh_body,veh_age,area,agecat,veh_value,exposure,claimcst0,frequincy,severity])
        x=pd.DataFrame([row],columns=columns)
        st.dataframe(x)


        make_prediction=model.predict(x)[0]
        Final_Result=""
        if make_prediction >0.5:
            Final_Result.append("Male")
        else:
            Final_Result.append("Female")
            
       
        st.subheader(f"the severity is {Final_Result}")


if __name__=="__main__" :
    show_predict_page()



