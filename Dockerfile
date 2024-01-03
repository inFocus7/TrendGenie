# Need C module to be compiled
FROM debian:12-slim AS builder
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes \
    python3-venv gcc libpython3-dev libjpeg-dev zlib1g-dev libfreetype6-dev && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

FROM builder as builder-venv
COPY requirements.txt requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

# Using the distroless image led to some issues.
# Without `/venv/bin/python3` `gradio` module was not found. With it, there were some UTF-8 encoding errors.
#FROM gcr.io/distroless/python3-debian12
FROM python:3.11
COPY --from=builder-venv /venv /venv
COPY . /app
WORKDIR /app
CMD ["/venv/bin/python3", "main.py"]

EXPOSE 7860