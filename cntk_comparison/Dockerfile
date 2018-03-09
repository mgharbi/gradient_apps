FROM microsoft/cntk:2.3-gpu-python3.5-cuda8.0-cudnn6.0

RUN apt-get update && apt-get install -y graphviz
RUN apt-get update && apt-get install -y cuda-command-line-tools-8-0

ADD code /code
ADD data /data

VOLUME /output

WORKDIR /code
CMD ./test.sh
