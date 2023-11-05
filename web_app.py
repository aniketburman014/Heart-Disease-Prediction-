import numpy as np
import pickle
import pandas as pd
import streamlit as st

rf_model=pickle.load(open('heart-disease-rf_model.pk1' , 'rb'))
scaler=pickle.load(open('standard-scaler.pkl', 'rb'))


# loading the useful columns names of dataset
loaded_data = pickle.load(open('datalists.pkl', 'rb'))
dataset_columns, continuous_feature, discrete_feature, scaled_columns=loaded_data




def operate(inputData):
    qf=pd.DataFrame(inputData,columns=dataset_columns)
    data_new=pd.get_dummies(qf,columns=discrete_feature)
    data_new[continuous_feature]=scaler.transform(data_new[continuous_feature])
    
    for column in scaled_columns:
        if column not in data_new:
           data_new[column] = 0
           data_new[[column]]=data_new[[column]]
        
    data_new=data_new[scaled_columns]
    
    return data_new
    





def HeartDiseasePredict(inputData):
    inputData=np.asarray(inputData)
    inputData=inputData.reshape(1, -1)
    dataset=operate(inputData)
    
    rf_pred=rf_model.predict(dataset)
    
    
    
    
    if(rf_pred[0]==0):
        return 'Person is not having heart disease'
    
    else:
        return 'Person may have heart disease'
    
    
    
    



st.title('Heart Disease Prediction ')
st.subheader('Please fill the below attributes :')
col1, col2, col3 = st.columns(3)
    
with col1:
    age = st.text_input('Age')
        
with col2:
     sex = st.text_input('Sex')
        
with col3:
     cp = st.text_input('Chest Pain types')
        
with col1:
     trestbps = st.text_input('Resting Blood Pressure')
        
with col2:
     chol = st.text_input('Serum Cholestoral in mg/dl')
        
with col3:
     fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
with col1:
     restecg = st.text_input('Resting Electrocardiographic results')
        
with col2:
     thalach = st.text_input('Maximum Heart Rate achieved')
        
with col3:
     exang = st.text_input('Exercise Induced Angina')
        
with col1:
     oldpeak = st.text_input('ST depression induced by exercise')
        
with col2:
     slope = st.text_input('Slope of the peak exercise ST segment')
        
with col3:
     ca = st.text_input('Major vessels colored by flourosopy')
        
with col1:
     thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
if st.button('Heart Disease Test Result'):
    pred=HeartDiseasePredict([age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]) 
    st.success(pred)
    
    

   
       
        
        
