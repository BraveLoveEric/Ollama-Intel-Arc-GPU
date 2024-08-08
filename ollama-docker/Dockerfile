FROM docker.kuku.me/python:3.11-slim

WORKDIR /code 

COPY ./requirements.txt ./
COPY ./ollama-linux-amd64 ./
COPY ./install.sh ./

RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN bash ./install.sh

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
