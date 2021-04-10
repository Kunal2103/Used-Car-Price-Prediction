#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from PIL import Image


# In[2]:


"""
@author: Krunal Pandya
"""


# In[3]:


st.title("Used-Car Price")


# In[4]:


html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Used Car Price-Prediction App </h2>
    </div>
"""
st.markdown(html_temp,unsafe_allow_html=True)


# In[7]:


image = Image.open('C:/cardata/used_Cars.jpg')

st.image(image, use_column_width='auto')


# In[8]:


def user_input_features():
    
        Owner_Type = st.selectbox('Owner_Type',('First','Second','Third','Fourth & Above'))
        Transmission = st.selectbox('Transmission',('Manual', 'Automatic'))
        Fuel_Type = st.selectbox('Fuel_Type',('Diesel', 'Electric', 'LPG','Petrol'))
        Location = st.selectbox('Location',('Bangalore','Chennai','Coimbatore','Mumbai','Hyderabad','Jaipur','Kochi','Kolkata','Delhi','Pune'))

        Kilometers_Driven = st.text_input("Kilometers_Driven",50000)
        Mileage = st.text_input("Mileage (kmpl)",15)
        Power = st.text_input("Power (bhp)", 50)
        Seats = st.slider('Seats',1,10,4)  
        years_old_from_2021 = st.slider('years_old_from_2021',1,4,2)  


        
        data = {
                'Owner_Type': Owner_Type,
                'Transmission' : Transmission,
                'Fuel_Type': Fuel_Type,
                'Location': Location,
                'Kilometers_Driven': Kilometers_Driven,
                'Mileage (kmpl)' : Mileage ,
                'Power (bhp)': Power ,
                'Seats' : Seats,
                'year_diff_from_2021' : years_old_from_2021}
        features = pd.DataFrame(data, index=[0])
        return features
     


# In[12]:


input_df = user_input_features()


# In[13]:


cars_new = pd.read_csv('C:/cardata/car_cleaned_data.csv')
df = pd.concat([input_df,cars_new],axis=0)


# In[14]:


encode = ['Fuel_Type','Location']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] 


# In[15]:


df.Transmission = df.Transmission.replace(['Automatic', 'Manual'],[1,0])
df.Owner_Type = df.Owner_Type.replace(['First', 'Second','Third','Fourth & Above'],[1,2,3,4])

df = df[:1]


# In[16]:


df['Kilometers_Driven'] = df['Kilometers_Driven'].astype('float')
df['Power (bhp)'] = df['Power (bhp)'].astype('float')


# In[19]:


df['Kilometers_Driven'] = np.log1p(df['Kilometers_Driven'])
df['Power (bhp)'] = np.log1p(df['Power (bhp)'])
df['year_diff_from_2021'] = np.log1p(df['year_diff_from_2021'])


# In[20]:


scaler = StandardScaler()                    

df[['Mileage (kmpl)']] = scaler.fit_transform(df[['Mileage (kmpl)']])    #Mileage follows normal distribution


# In[22]:


#Reads in saved Regression model
load_model = pickle.load(open('C:\cardata\Carprice_Model.pkl', 'rb'))


# In[23]:


result=""

if st.button("Predict Price"):
        result= float('%.2f'%np.exp(load_model.predict(df))) 
st.success('The Price is {} Lacs'.format(result))

