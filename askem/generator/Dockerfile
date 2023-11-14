# Generator Dockerfile
# this is fully self-contained, no need to install askem library
FROM huggingface/transformers-pytorch-gpu:4.29.2

RUN apt-get update && apt-get install -y \
    curl
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

EXPOSE 4503

HEALTHCHECK CMD curl --fail http://localhost:4503
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4503"]
