FROM huggingface/transformers-pytorch-deepspeed-latest-gpu:latest

WORKDIR /askem

# Install dependencies
COPY ./.devcontainer/requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY ./askem ./askem
COPY ./train.py .

ENTRYPOINT ["python", "train.py"]
