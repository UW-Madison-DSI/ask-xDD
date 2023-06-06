# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY askem/demo/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./askem/demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]