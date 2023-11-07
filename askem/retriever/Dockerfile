FROM python:3.11.6-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Serving
EXPOSE 4502
ENTRYPOINT ["uvicorn", "--app-dir=/app", "app:app", "--host", "0.0.0.0", "--port", "4502"]
