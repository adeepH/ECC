import streamlit as st
import boto3
import joblib
import pandas as pd
import tarfile
from infer import *

# S3 bucket and key where the model is stored
BUCKET_NAME = 'sagemaker-us-east-1-686953405862'
MODEL_KEY = 'output_1681940773/Experiment-1681940777891/data-processor-models/Experiment-1681940777891-dpp0-1-8dc6e39238ce47a4b3a71b5e9e4a93b/output/model.tar.gz'
access_key = 'AKIAZ74M5DWTMEJQ7J5T'
secret_key = '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'


# Load the trained model from S3

s3 = boto3.client('s3', aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key) 

files = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)

st.write(joblib.load(files))
