#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction
# #### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stat
from sklearn.impute import KNNImputer
import pylab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from sklearn.metrics import r2_score , mean_squared_error
import streamlit as st
import pickle
from PIL import Image


# In[2]:


#load data
df = pd.read_csv('car_data.csv')


# In[3]:


#head
df.head()


# In[4]:


#shape
df.shape


# In[5]:


#set car-name as index
df = df.set_index('Name')


# #### Data Pre-processing on Mileage, Engine, Power features to convert it into numeric

# In[6]:


df['Engine']  = df['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))
df['Power']   = df['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))
df['Mileage'] = df['Mileage'].apply(lambda x: str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))
df['Mileage'] = df['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))


# In[7]:


#renaming columns
df = df.rename(columns={'Engine': 'Engine (CC)','Power': 'Power (bhp)', 'Mileage': 'Mileage (kmpl)'})


# In[8]:


df.head()


# In[9]:


df.dtypes


# In[10]:


df['Power (bhp)'] = df['Power (bhp)'].replace('null ', np.NaN)


# In[11]:


df['Power (bhp)'] = df['Power (bhp)'].astype('float')
df['Engine (CC)'] = df['Engine (CC)'].astype('float')
df['Mileage (kmpl)'] = df['Mileage (kmpl)'].astype('float')


# In[12]:


year = 2021

df['year_diff_from_2021'] = year - df['Year']
df.drop('Year',axis = 1, inplace = True)


# In[13]:


df.head()


# ## Feature Engineering
# ### missing value analysis

# In[14]:


df.isnull().sum()


# In[15]:


#imputing missing values
imputer = KNNImputer()
df[['Mileage (kmpl)','Engine (CC)','Power (bhp)']] = imputer.fit_transform(df[['Mileage (kmpl)','Engine (CC)','Power (bhp)']])

df.Seats = df.Seats.fillna(df.Seats.mode()[0])


# In[16]:


df.isnull().sum()


# In[17]:


plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
sns.distplot(df['Engine (CC)'])

plt.subplot(2,3,2)
sns.distplot(df['Power (bhp)'])

plt.subplot(2,3,3)
sns.distplot(df['Kilometers_Driven'])

plt.subplot(2,3,4)
sns.distplot(df['Mileage (kmpl)'])

plt.subplot(2,3,5)
sns.distplot(df['year_diff_from_2021'])

plt.subplot(2,3,6)
sns.distplot(df.Price)


# * As we can see fom distplot, data is not normally distributed it is skewed so , Transformation is required to convert it into a normal distribution except 'Mileage (kmpl)' feature. It follows normal distribution. we will treat the outliers present in this feature later.
# ### we will use feature transformation technique [logarithmic transformation]

# In[18]:


df['Engine (CC)'] = np.log1p(df['Engine (CC)'])
df['Power (bhp)'] = np.log1p(df['Power (bhp)'])
df['Kilometers_Driven'] = np.log1p(df['Kilometers_Driven'])
df['year_diff_from_2021'] = np.log1p(df['year_diff_from_2021'])
df['Price'] = np.log(df['Price'])


# In[19]:


def plot_data(df,feature):
    plt.figure(figsize=(8,5))
    plt.subplot(1,2,1)
    sns.distplot(df[feature])
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()


# In[20]:


plot_data(df, 'Engine (CC)')


# In[21]:


plot_data(df , 'Power (bhp)')


# In[22]:


plot_data(df, 'Kilometers_Driven')


# In[23]:


plot_data(df,'year_diff_from_2021')


# In[24]:


plot_data(df,'Price')


# ### let's remove outliers present in Mileage (kmpl) column
# * For any car, mileage can not be 0 , so removing it

# In[25]:


df = df.drop(df[df['Mileage (kmpl)']== 0].index, axis = 0)


# In[26]:


sns.distplot(df['Mileage (kmpl)'])


# ### Label Encoder for Ordinal Categories

# In[27]:


df.Transmission.value_counts()


# In[28]:


df.Owner_Type.value_counts()


# In[29]:


le = LabelEncoder()
df[['Transmission','Owner_Type']] = df[['Transmission','Owner_Type']].apply(le.fit_transform)


# ### dummy variables for nominal categories

# In[30]:


df.head()


# In[31]:


df = pd.get_dummies(df,columns = ['Fuel_Type','Location'],drop_first=True)


# ### Dataset is now ready for the Model building
# #### train-test split

# In[32]:


X = df.drop('Price',axis = 1)             #predictor variables
y = df.Price                              #target variable


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=21)

X_train.shape , X_test.shape , y_train.shape , y_test.shape


# ### Feature Selection

# In[34]:


corr = X_train.corr()
plt.figure(figsize=(22,12))
sns.heatmap(corr,annot = True,cmap = 'coolwarm')


# * Power (bhp) and Engine (CC) are highly positively correlated, so removing one of them #### (thresold > 0.8)

# In[35]:


X_train.drop('Engine (CC)',axis = 1, inplace=True)
X_test.drop('Engine (CC)',axis = 1, inplace=True)


# ## Feature Scaling
# ### Standardization

# In[36]:


scaler = StandardScaler()                    

X_train[['Mileage (kmpl)']] = scaler.fit_transform(X_train[['Mileage (kmpl)']])    #Mileage follows normal distribution

X_test[['Mileage (kmpl)']] = scaler.transform(X_test[['Mileage (kmpl)']])


# ## Model Building
# ### Random Forest

# In[37]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=250,max_features='auto',random_state=21).fit(X_train, y_train)


# In[38]:


pred  = rf.predict(X_test)


# In[39]:


print('Accuracy =', r2_score(y_test,pred)*100)


# In[40]:


print('MSE =',mean_squared_error(y_test, pred))


# In[41]:


#taking exp to predicted data so that we get it in a actual form
final_price = np.exp(pred)


# In[42]:


final_price[:25]


# ### Model Evaluation

# In[43]:


pred_train = rf.predict(X_train)


# In[44]:


print('Accuracy on train data =' , r2_score(y_train,pred_train)*100)


# #### Model works well on train and test data, No overfitting problem

# # Model Deployment

# In[45]:


"""
Created by: Krunal Pandya
"""


# In[46]:


st.sidebar.title("Used-Car Price")


# In[47]:


st.sidebar.write("""
This app predicts the **price** of the used cars. Get your car price based on your requirements.
""")


# In[48]:


html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Used Car Price-Prediction App </h2>
    </div>
"""
st.markdown(html_temp,unsafe_allow_html=True)


# In[49]:


image = Image.open('C:/cardata/used_Cars.jpg')

st.sidebar.image(image, use_column_width='auto')


# In[50]:


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
     


# In[51]:


input_df = user_input_features()


# In[52]:


cars_new = pd.read_csv('C:/cardata/car_cleaned_data.csv')
df = pd.concat([input_df,cars_new],axis=0)


# In[54]:


encode = ['Fuel_Type','Location']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] 


# In[55]:


df.Transmission = df.Transmission.replace(['Automatic', 'Manual'],[1,0])
df.Owner_Type = df.Owner_Type.replace(['First', 'Second','Third','Fourth & Above'],[1,2,3,4])

df = df[:1]


# In[56]:


df['Kilometers_Driven'] = df['Kilometers_Driven'].astype('float')
df['Power (bhp)'] = df['Power (bhp)'].astype('float')


# In[57]:


df['Kilometers_Driven'] = np.log1p(df['Kilometers_Driven'])
df['Power (bhp)'] = np.log1p(df['Power (bhp)'])
df['year_diff_from_2021'] = np.log1p(df['year_diff_from_2021'])


# In[59]:


scaler = StandardScaler()                    

df[['Mileage (kmpl)']] = scaler.fit_transform(df[['Mileage (kmpl)']])    #Mileage follows normal distribution


# In[62]:


result=""

if st.button("Predict Price"):
        result= float('%.2f'%np.exp(rf.predict(df))) 
st.success('The Price in Lacs: {}'.format(result))

