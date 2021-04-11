#!/usr/bin/env python
# coding: utf-8

# In[35]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split


# In[45]:


f1 = pd.read_csv('C:/cardata/new_Car.csv')


# In[46]:


X = f1.drop('Price',axis = 1)             #predictor variables
y = f1.Price                              #target variable


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=21)


# In[48]:


X_train.drop('Engine (CC)',axis = 1, inplace=True)
X_test.drop('Engine (CC)',axis = 1, inplace=True)


# In[49]:


scaler = StandardScaler()                    

X_train[['Mileage (kmpl)']] = scaler.fit_transform(X_train[['Mileage (kmpl)']])    #Mileage follows normal distribution

X_test[['Mileage (kmpl)']] = scaler.transform(X_test[['Mileage (kmpl)']])


# In[39]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=250,max_features='auto',random_state=21).fit(X_train, y_train)


# In[27]:


"""
Created by: Krunal Pandya
"""


# In[11]:


st.sidebar.title("Used-Car Price")


# In[28]:


st.sidebar.write("""
This app predicts the **price** of the used cars. Get your car price based on your requirements.
""")


# In[12]:


html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Used Car Price-Prediction App </h2>
    </div>
"""
st.markdown(html_temp,unsafe_allow_html=True)


# In[13]:


image = Image.open('C:/cardata/used_Cars.jpg')

st.sidebar.image(image, use_column_width='auto')


# In[14]:


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
     


# In[15]:


input_df = user_input_features()


# In[16]:


cars_new = pd.read_csv('C:/cardata/car_cleaned_data.csv')
df = pd.concat([input_df,cars_new],axis=0)


# In[17]:


encode = ['Fuel_Type','Location']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] 


# In[18]:


df.Transmission = df.Transmission.replace(['Automatic', 'Manual'],[1,0])
df.Owner_Type = df.Owner_Type.replace(['First', 'Second','Third','Fourth & Above'],[1,2,3,4])

df = df[:1]


# In[19]:


df['Kilometers_Driven'] = df['Kilometers_Driven'].astype('float')
df['Power (bhp)'] = df['Power (bhp)'].astype('float')


# In[20]:


df['Kilometers_Driven'] = np.log1p(df['Kilometers_Driven'])
df['Power (bhp)'] = np.log1p(df['Power (bhp)'])
df['year_diff_from_2021'] = np.log1p(df['year_diff_from_2021'])


# In[21]:


scaler = StandardScaler()                    

df[['Mileage (kmpl)']] = scaler.fit_transform(df[['Mileage (kmpl)']])    #Mileage follows normal distribution


# In[42]:


result=""

if st.button("Predict Price"):
        result= float('%.2f'%np.exp(rf.predict(df))) 
st.success('The Price in Lacs: {}'.format(result))

