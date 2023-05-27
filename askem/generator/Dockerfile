# This is the generator Dockerfile
FROM huggingface/transformers-pytorch-gpu:4.29.2

RUN pip install fastapi pydantic uvicorn

EXPOSE 4503

WORKDIR /app

COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4503"]