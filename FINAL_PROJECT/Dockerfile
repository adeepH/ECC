# pulling an image
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#PORT
#EXPOSE 8501

# CORS will not recognize certain url
CMD ["streamlit", "run", "--server.enableCORS", "false", "app.py"]
