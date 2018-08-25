FROM bids/base_fsl

MAINTAINER Sajjad Torabian <torabiansajjad@gmail.com>

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    python2.7
    apt-get remove --purge fuse && \
    apt-get remove --purge python-fuse && \

RUN apt-get update && \
    apt-get install -y python-mvpa2

RUN mkdir -p /code

COPY run.py /code/run.py
RUN chmod +x /code/run.py

COPY version /code/version

ENTRYPOINT ["/code/run.py"]
