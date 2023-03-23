import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import load_model

st.title('Cutting Model')

Feed_rate = st.number_input('Feed Rate')
Depth_of_cut = st.number_input('Depth of Cut')
Ultrasonic_Vibration = st.selectbox('Ultrasonic Vibration', ['Yes', 'No'])
Cutting_Fluid = st.selectbox('Cutting Fluid', ['Yes', 'No'])

scaler = pickle.load(open('scaler.pkl','rb'))

prediction_array = scaler.transform(np.array([[Feed_rate, Depth_of_cut]]))

for n in [Ultrasonic_Vibration, Cutting_Fluid]:
    if n == "Yes":
        n = 1
    else:
        n = 0   
    prediction_array = np.append(prediction_array,[[n]])

model = load_model('ANN_model.h5')
Force = np.exp(model.predict(prediction_array.reshape(1,4))[0,0])
Sur_rough = np.exp(model.predict(prediction_array.reshape(1,4))[0,1])

if st.button('Predict'):
    if (np.round(scaler.inverse_transform(prediction_array[:2].reshape(1,-1))[0][0],4)==0) | (np.round(scaler.inverse_transform(prediction_array[:2].reshape(1,-1))[0][1],4)==0):
        st.header('Missing Data')
    else:
        st.header(':Cutting Force = :green[' + str(np.round(Force,3))+' N]')
        st.header('Surface Roughness = :green[' + str(np.round(Sur_rough,3))+' µm ]')