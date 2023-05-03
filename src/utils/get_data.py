import boto3
import pandas as pd
access_key = 'AKIAZ74M5DWTMEJQ7J5T'
secret_key = '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'
s3 = boto3.client('s3', aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)

bucket_name = 'sagemaker-us-east-1-686953405862'
file_name = 'frauddata.csv' # or 'YOUR_FILE_NAME.xlsx' for Excel files
response = s3.get_object(Bucket=bucket_name, Key=file_name)
temp_file = 'data.csv'
print(True)
s3.download_file(bucket_name, file_name, temp_file)
print(True)
df = pd.read_csv(temp_file) # or pd.read_excel(response['Body'])
print(True)
print(df.head(10))

import os

os.remove(temp_file)