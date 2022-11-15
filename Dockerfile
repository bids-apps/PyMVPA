# Use Ubuntu 18.04 LTS (compatible with python-mvpa2 package)
FROM ubuntu:18.04

MAINTAINER Sajjad Torabian <torabiansajjad@gmail.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends python2.7

RUN apt-get update && \
    mkdir /dev/fuse && \
    chmod 777 /dev/fuse && \
    apt-get install -y python-mvpa2 && \
    apt-get remove -f -y --purge fuse

RUN mkdir -p /code

COPY run.py /code/run.py
RUN chmod +x /code/run.py

COPY version /code/version

ENTRYPOINT ["/code/run.py"]
