FROM microsoft/cntk:2.5.1-gpu-python3.5-cuda9.0-cudnn7.0
# FROM microsoft/cntk:2.5.1-cpu-python3.5

RUN apt-get update && apt-get install -y graphviz
RUN apt-get update && apt-get install -y cuda-command-line-tools-9-0

ADD code /code
ADD data /data

VOLUME /output

WORKDIR /code
CMD ./test.sh
