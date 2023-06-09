FROM ubuntu:22.04

ADD . /app/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    openjdk-11-jre-headless \
    gcc \
    gfortran \
    gdal-bin \
    libgdal-dev \
    python3-pip
    
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

RUN export JAVA_HOME

RUN pip3 install -r requirements-dev.txt

RUN make