FROM python:3.8-slim-buster

# Update packages and install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a directory for the app code
RUN mkdir /app
WORKDIR /app

# Copy the requirements file to the container and install the necessary packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code to the container
COPY login.py app/
COPY pages/ /app/
# Expose the port used by Streamlit
EXPOSE 8502
HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health
WORKDIR /dapp
# Start the Streamlit app
#CMD ["streamlit", "run", "--server.port", "8502", "app/login.py"]

# alternative
ENTRYPOINT ["streamlit", "run"]
CMD ["login.py", "--server.maxUploadSize", "500"]