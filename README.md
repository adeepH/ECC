## Fraud Detection Portal: A Cloud-based Service

This repository contains code and documentation for the Fraud Detection Portal, a cloud-based service for detecting fraudulent activity in banking systems. The system uses machine learning and deep learning algorithms to predict potential fraudulent customers/transactions in the future.

## Getting Started

To get started trying to recreate the application:
, follow these steps:

1. Clone the branch ``Fraud-detection-portal`` of the repository to your local machine using ```git clone https://github.com/adeepH/ECC.git```.
2. Install all the required packages by running the command ```pip install -r requirements.txt```.
2. Change the ```Access Key ID and Secret Password (obtained from S3)``` in ```pages/infer.py```.
3. After install the required packages, run ```streamlit run login.py``` to run the web application locally.

## Repository Structure

The repository is organized as follows:

```README.md```: This file provides an overview of the repository and instructions for getting started with the assignments.

1. ```models/```: This folder contains model file as a pickle object.
2. ```pages/```: This folder contains the inference page, and is structured in such a way to enable multipage application on streamlit.
3. ```src/```: This folder contains all the codes and relevant data to our study.

## Contributing
- Adeep Hande (ahande@iu.edu)
- Kumar Vinayak (kvinayak@iu.edug)
- Dene Absher (dabsher@iu.edu)

## License
This repository is licensed under the MIT License.

