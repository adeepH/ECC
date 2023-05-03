import streamlit as st
import time
import altair as alt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from io import BytesIO
import seaborn as sns
import plotly.graph_objects as go 
import boto3
import pickle
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')
access_key = 'AKIAZ74M5DWTMEJQ7J5T'
secret_key = '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'

st.set_option('deprecation.showPyplotGlobalUse', False)

# Increase maximum file size to 500 MB
# st.set_option('server.maxUploadSize', 500)
# adding it in command line

# Define the Streamlit app
def app():
    # Set page title
#CSS
    custom_css = """
    <style>
        .fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background-color: #f0f2f6;
        }
        h1, h2, h3, h4, h5, h6, label,button,title{
            font-family: "Copperplate", Copperplate;
            color: #00FF33;
            margin-top: 100;
        }
        .stButton>button {
            background-color: #1a202c;
            color: #f0f2f6;
            font-weight: bold;
        }
    </style>
    """
    # Set page title
    image = Image.open("data/Banking-logo-by-Friendesign-Acongraphic-10-580x386.jpg")
    # Resize the image
    new_width = 400  # You can change this value to your preferred width
    new_height = int(new_width * image.height / image.width)
    image = image.resize((new_width, new_height))
    col1, col2, col3 = st.columns([8.8, 9, 8.8])
    with col2:
        st.image(image)
    #st.image(image)
     

    if 'user_dict' not in st.session_state:
        st.warning('Please Log in before accessing the fraud detection page')
        st.stop()
    users = st.session_state.user_dict
    #st.image(image, use_column_width=True)
    if st.session_state.username in users and st.session_state.password in users[st.session_state.username]:

        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown('<div class="fullscreen"></div>', unsafe_allow_html=True)
        with st.container():
            # Display title and description
            col1, col2 = st.columns([1.5, 3])
            with col2:
                st.title('Fraud Detection')
            
            st.markdown('Upload a CSV file with the input data and click "Run Inference" to predict the target variable using a Random Forest Classifier.')
            st.write('Upload a CSV file:')
            uploaded_file = st.file_uploader('', type='csv')
        fnames = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
        # Train the Random Forest Classifier when user clicks "Train Model"
        if st.button('Infer'):
            # Check if file was uploaded
            if uploaded_file is not None:
                # Load the uploaded file into a dataframe
                df = pd.read_csv(uploaded_file)
                #st.write('Reading the dataframe')
                # Check if the dataframe has the required columns
                if set(fnames).issubset(set(df.columns)):
                    #st.write('Done preliminary Checks')
                    # Train a Random Forest Classifier on the dataframe
                    #df = preprocess(df) # Do some preprocessing
                    #st.write('Preprocessed the data')
                    #x_train,x_test,y_train,y_test = train_test(df=df)
                    #st.write('Split the data') 
                    

                    #pred, score = XGBoost_clf(x_train,x_test,y_train,y_test)
                    #push_to_s3(x_train, y_train)
                    
                    model = load_s3_artifacts()
                    y_test = df['isFraud']
                    x_test = df.drop(columns='isFraud')
                    pred = model.predict(x_test) 
                    print(pred)
                    #score = accuracy_score(y_test, pred)
                    
                    #st.success('Found Results')
                    #st.write(score)
                    x_test['isFraud'] = pred
                    #st.write(x_test['isFraud'].value_counts())
                    x_test['isFraud'] = x_test['isFraud'].apply({1:'Fraud', 0:'Fair'}.get)
                    visualize(x_test)
                    st.write('Showing a sample of the predictions')
                    st.dataframe(x_test.head(20))

                    # Add download button
                    button = st.download_button(
                        label="Download results",
                        data=df.to_csv(index=0).encode('utf-8'),
                        file_name='data.csv',
                        mime='text/csv'
                    )
                    #st.write(st.dataframe(pred))

                    if button:
                        st.success('File downloaded successfully!!s')
                else:
                    st.error("CSV file must contain columns 'step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'")
            else:
                st.error('Please upload a CSV file.')
    else:
        st.warning('Please head over the login page to log in')
    

def preprocess(df):
    
    df = df.dropna()
    df['type']=df['type'].replace({'CASH_OUT':1,'PAYMENT':2,'CASH_IN':3,'TRANSFER':4,'DEBIT':5})
    return df

@st.cache_resource # save the train, test files in streamlit cache
def train_test(df):

    X=df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]
    y=df[['isFraud']]

    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    test = pd.concat([x_test, y_test], axis=1)
    test.to_csv('test.csv', index=0)
    st.write(x_train.shape)
    st.write(x_test.shape)
    return x_train,x_test,y_train,y_test

@st.cache_resource # save the outputs in streamlit cache
def RF_Classifier(x_train,x_test,y_train,y_test):

    rf=RandomForestClassifier()

    rf_model=rf.fit(x_train,np.ravel(y_train))


    pred = rf_model.predict(x_test)
    score = rf_model.score(y_test, pred)

    return pred, score

#@st.cache_resource # save the outputs in streamlit cache
def XGBoost_clf(x_train, x_test, y_train, y_test):
    XGB = xgb.XGBClassifier()
    XGB.fit(x_train, np.ravel(y_train)) 
    #n_features = len(XGB.feature_importances_)
    #st.write(f'Trained model has {n_features} features')
    #st.write('I have fit the model')
    st.write(x_test.shape)
    st.write(XGB)
    pred = XGB.predict(x_test)
    score = accuracy_score(y_test, pred)


    return pred, score

def push_to_s3(x_train, y_train):
    
    XGB = xgb.XGBClassifier()
    XGB.fit(x_train, np.ravel(y_train))
    # Save model as pickle file
    with open('model.pkl', 'wb') as f:
        pickle.dump(XGB, f)
    #pickle_bytes = pickle.dumps(model)
    s3 = boto3.resource('s3', aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)

    bucket_name = 'sagemaker-us-east-1-686953405862'    
    bucket = s3.Bucket(bucket_name)
    key = 'model.pkl'
    with open('model.pkl', 'rb') as f:
        s3.Bucket(bucket_name).upload_fileobj(f, key)
    #bucket.put_object(Key='model.pkl', Body=pickle_bytes)

@st.cache_resource()
def load_s3_artifacts():

    s3 = boto3.resource('s3', aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)
    bucket_name = 'sagemaker-us-east-1-686953405862'
    file_name = 'model.pkl' # or 'YOUR_FILE_NAME.xlsx' for Excel files
    #response = s3.get_object(Bucket=bucket_name, Key=file_name)
    temp_file = 'model.pkl' 
    with open('model.pkl', 'wb') as f:
        s3.Bucket(bucket_name).download_fileobj(file_name, f)
    #s3.download_file(bucket_name, file_name, temp_file) 
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    #    model = pickle.load(f['body'])
    success_message = st.success('Loaded the pretrained Model')
    time.sleep(5)
    success_message.empty()
    st.write(model)
    return model

def visualize(df):
# Specify binary variable
    labels = ['Fraud', 'Fair']
    values = [df['isFraud'].value_counts()[0], df['isFraud'].value_counts()[1]]  # These values should add up to 100

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='percent', hole=.3)])
    st.plotly_chart(fig)



if __name__ == "__main__":
    
    app()