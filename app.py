import streamlit as st
import pickle,gzip
import numpy as np


st.title('Thyroid Prediction App')

age = st.slider( 'Input the age',  min_value = 0	, max_value=80) 
#'F':1, 'M':0
sex = st.selectbox( 'Select Gender',  ('Male','Female'))  
#{"t":1,"f":0}
on_thyroxine = st.selectbox(' On thyroxine ',  ('Yes','No')	) 
#{"y":1, "n":0}
query_hypothyroid = st.selectbox(' Examined in past-hypothyroid ',  ('Yes','No')	) 
query_hyperthyroid = st.selectbox(' Examined in past-hyperthyroid',  ('Yes','No')	) 
psych = st.selectbox(' psych',  ('Yes','No')	) 
TSH_measured = st.selectbox(' TSH_measured',  ('Yes','No')	) 
TSH = st.slider( 'TSH',  min_value = -345.2582345639	, max_value = 880.7955352204)     
T3_measured = st.selectbox(' T3_measured',  ('Yes','No')	)
T3 = st.slider( 'T3',  min_value = 0	, max_value = 18)     
TT4_measured = st.selectbox(' TT4_measured',  ('Yes','No')	)
TT4 = st.slider( 'TT4',  min_value = -29.0828049132	, max_value = 600.0)     
T4U_measured = st.selectbox(' T4U_measured',  ('Yes','No')	)
FTI_measured = st.selectbox(' FTI_measured',  ('Yes','No')	)
FTI = st.slider( 'FTI',  min_value = -241.5120614242	, max_value = 881.0)     

submit = st.button('Predict')
if submit:
    #encoding
    d_sex= {'Male':0, 'Female':1}
    d_YesNo= {'Yes':1,'No':0}
    input_list =[]


    input_list.append(age)
    input_list.append(d_sex[sex])
    input_list.append(d_YesNo[on_thyroxine])
    input_list.append(d_YesNo[query_hypothyroid])
    input_list.append(d_YesNo[query_hyperthyroid])
    input_list.append(d_YesNo[psych])
    input_list.append(d_YesNo[TSH_measured])
    input_list.append(TSH)
    input_list.append(d_YesNo[T3_measured])
    input_list.append(T3)
    input_list.append(d_YesNo[TT4_measured])
    input_list.append(TT4)
    input_list.append(d_YesNo[T4U_measured])
    input_list.append(d_YesNo[FTI_measured])
    input_list.append(FTI)
    


    
    print(input_list)

    #scaling
    scaled_input = pickle.load(open("saved_scaler_models\thyroid_scaler.pickle", "rb"))        
    
    input_list_reshaped = np.array(scaled_input).reshape(1,-1)
    
    
    #loading the stored model for prediction
    model = pickle.load(gzip.open('saved_scaler_models\thyroid_model.pkl', 'rb'), encoding ='latin1' )
    output = model.predict(input_list_reshaped)
    if output:
        st.write(output)