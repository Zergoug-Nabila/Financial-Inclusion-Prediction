import streamlit as st
import pandas as pd
import pickle

# Title of the web app
st.title('Financial Inclusion Prediction')

st.write("""
### Dataset description: 
    The dataset contains demographic information and what financial services are 
    used by approximately 33,600 individuals across East Africa. The ML model 
    role is to predict which individuals are most likely to have or use a bank 
    account.
""")

# Function to collect user input
def user_input_features():
    country = st.selectbox('country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
    year = st.slider('year', 2016, 2018, 2017)
    location_type = st.radio('location_type', ['Rural', 'Urban'])
    cellphone_access = st.radio('cellphone_access', ['Yes', 'No'])
    household_size = st.slider('household_size', 1, 21, 8)
    age_of_respondent = st.slider('age_of_respondent', 16, 100, 24)
    gender_of_respondent = st.radio('gender_of_respondent', ['Male', 'Female'])
    relationship_with_head = st.selectbox('relationship_with_head', ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives'])
    marital_status = st.selectbox('marital_status', ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Separated', 'Don\'t know'])
    education_level = st.selectbox('education_level', ['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'])
    job_type = st.selectbox('job_type', ['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])

    input_data = {
        'country': country,
        'year': year,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size': household_size,
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': gender_of_respondent,
        'relationship_with_head': relationship_with_head,
        'marital_status': marital_status,
        'education_level': education_level,
        'job_type': job_type
    }
    data = pd.DataFrame(input_data, index=[0])

    return data

input_data = user_input_features()

# Function to encode features (replace with your actual encoding logic)
def encode_features(df):
    region_mapping = {
        'Kenya': 0,
        'Rwanda': 1,
        'Tanzania': 2,
        'Uganda': 3
    }

    location_type_mapping = { 
        'Rural':0, 
        'Urban':1
    }

    cellphone_access_mapping = {
        'Yes':1, 
        'No':0
    }

    gender_of_respondent_mapping = {
        'Male':0, 
        'Female':1
    }

    relationship_with_head_mapping = {'Spouse':0, 'Head of Household':1, 'Other relative':2, 'Child':3, 'Parent':4,
       'Other non-relatives':5
    }

    marital_status_mapping = {'Married/Living together':0, 'Widowed':1, 'Single/Never Married':2,
       'Divorced/Seperated':3, 'Dont know':4
       }

    education_level_mapping = {'Secondary education':0, 'No formal education':1,
       'Vocational/Specialised training':2, 'Primary education':3,
       'Tertiary education':4, 'Other/Dont know/RTA':5
    }
    
    job_type_mapping = {'Self employed':0, 'Government Dependent':1,
       'Formally employed Private':2, 'Informally employed':3,
       'Formally employed Government':4, 'Farming and Fishing':5,
       'Remittance Dependent':6, 'Other Income':7,
       'Dont Know/Refuse to answer':8, 'No Income':9
    }
    
    df['country'] = df['country'].map(region_mapping)
    df['location_type'] = df['location_type'].map(location_type_mapping )
    df['cellphone_access'] = df['cellphone_access'].map(cellphone_access_mapping)
    df['gender_of_respondent'] = df['gender_of_respondent'].map(gender_of_respondent_mapping)
    df['relationship_with_head'] = df['relationship_with_head'].map(relationship_with_head_mapping)
    df['marital_status'] = df['marital_status'].map(marital_status_mapping)
    df['education_level'] = df['education_level'].map(education_level_mapping)
    df['job_type'] = df['job_type'].map(job_type_mapping)
    
    return df

# Encode the input features
input_df = encode_features(input_data)

# Load the trained model
with open('bank_account_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict using the trained model
def predict(model, input_df):
    prediction = model.predict(input_df)
    return prediction

# Prediction button
if st.button('Predict'):
    prediction = predict(model, input_df)
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Bank Account Ownership: Yes')
    else:
        st.write('Bank Account Ownership: No')
