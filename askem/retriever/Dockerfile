FROM python:3.9

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 4502

ENTRYPOINT ["uvicorn", "--app-dir=/app", "app:app", "--host", "0.0.0.0", "--port", "4502"]