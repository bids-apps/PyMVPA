FROM bids/base_fsl

MAINTAINER Sajjad Torabian <torabiansajjad@gmail.com>

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends python2.7

RUN sudo apt-get update && \
    mkdir /dev/fuse && \
    chmod 777 /dev/fuse && \
    sudo apt-get install -y python-mvpa2 && \
    sudo apt-get remove -f -y --purge fuse

RUN mkdir -p /code

COPY run.py /code/run.py
RUN chmod +x /code/run.py

COPY version /code/version

ENTRYPOINT ["/code/run.py"]
