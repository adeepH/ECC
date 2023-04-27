import streamlit as st
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError
import joblib
from sklearn.metrics import accuracy_score

# Define S3 credentials
S3_BUCKET = 'sagemaker-us-east-1-686953405862'
S3_ACCESS_KEY = 'AKIAZ74M5DWTMEJQ7J5T'
S3_SECRET_KEY = 'RPYZIqqRKeFF7o0Ch20MIu6aycIDXAoAVaNCvh1y'
file_name = 'frauddata.csv' # or 'YOUR_FILE_NAME.xlsx' for Excel files

# Connect to S3
s3 = boto3.client('s3',
                  aws_access_key_id=S3_ACCESS_KEY,
                  aws_secret_access_key=S3_SECRET_KEY)

# Define a function to download a file from S3
def download_s3_file(remote_path, local_path):
    try:
        s3.download_file(S3_BUCKET, remote_path, local_path)
        st.success(f'{remote_path} downloaded to {local_path}')
    except NoCredentialsError:
        st.error('Credentials not available')
    except Exception as e:
        st.error(f'Error downloading file: {e}')

# Define a function to load a model from a joblib file
def load_model(model_file):
    return joblib.load(model_file)

# Define a function to run inference on a dataframe using a model
def run_inference(model, df):
    # Replace with your own inference logic
    y_pred = model.predict(df)
    return y_pred

# Define the Streamlit app
def app():
    # Set page title
    st.set_page_config(page_title='AutoML Model Inference App')

    # Display title and description
    st.title('AutoML Model Inference App')
    st.write('Select a model from the dropdown, upload a CSV file, and click "Run Inference" to predict the target variable.')

    # Define model options
    model_options = [
        'model_1.joblib',
        'model_2.joblib',
        'model_3.joblib'
    ]

    # Display model dropdown
    model_file = st.selectbox('Select a model:', model_options)

    # Download the selected model from S3
    download_s3_file(f'automl-experiment/{model_file}', model_file)

    # Load the selected model
    model = load_model(model_file)

    # Display file upload widget
    st.write('Upload a CSV file:')
    uploaded_file = st.file_uploader('', type='csv')

    # Run inference when user clicks "Run Inference"
    if st.button('Run Inference'):

        # Check if file was uploaded
        if uploaded_file is not None:
            # Load the uploaded file into a dataframe
            df = pd.read_csv(uploaded_file)

            # Check if the dataframe has the required columns
            if {'feature_1', 'feature_2', 'target'} <= set(df.columns):
                # Run inference and display the results
                y_pred = run_inference(model, df[['feature_1', 'feature_2']])
                accuracy = accuracy_score(df['target'], y_pred)
                st.write(f'Predicted target values: {y_pred}')
                st.write(f'Accuracy: {accuracy}')
            else:
                st.error('CSV file must contain columns "feature_1", "feature_2", and "target".')
        else:
            st.error('Please upload a CSV file.')

app()