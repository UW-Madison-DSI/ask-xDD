FROM huggingface/transformers-pytorch-deepspeed-latest-gpu:latest

WORKDIR /askem

COPY ./.devcontainer/requirements.txt .
RUN pip install -r requirements.txt

COPY ./askem ./askem
COPY ./train.py .
COPY ./working.ipynb .

# Configure VSCode server
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz
RUN rm vscode_cli.tar.gz
ENTRYPOINT [ "./code",  "tunnel", "--accept-server-license-terms"]
