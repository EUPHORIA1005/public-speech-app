FROM python:3.10-slim

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "02_faster_whisper_pipeline.py"]
