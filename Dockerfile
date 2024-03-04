FROM ubuntu:20.04

COPY requirements.txt .
COPY ./app /app

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

RUN python3 -m nltk.downloader punkt wordnet omw-1.4

ENV PYTHONPATH = /app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]