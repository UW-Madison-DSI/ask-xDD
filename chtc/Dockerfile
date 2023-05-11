FROM huggingface/transformers-pytorch-deepspeed-latest-gpu:latest

WORKDIR /askem

COPY ./.devcontainer/requirements.txt .

RUN pip install -r requirements.txt

COPY ./askem ./askem

COPY ./train.py .

COPY ./working.ipynb . 