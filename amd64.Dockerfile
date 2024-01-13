# Need C module to be compiled
FROM debian:12-slim AS builder
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes \
    python3-venv gcc libpython3-dev libjpeg-dev zlib1g-dev libfreetype6-dev libsndfile1-dev fontconfig ffmpeg && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Install dependencies. This is done in a separate step to take advantage of Docker's caching, so that we don't have to
# reinstall dependencies every time we change the code.
FROM builder as builder-venv
COPY requirements.txt requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

FROM gcr.io/distroless/python3-debian12
# Copy over dependencies needed needed for Pillow to work.
COPY --from=builder /usr/lib/x86_64-linux-gnu/libjpeg.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libfreetype.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpng16.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libbrotlidec.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libbrotlicommon.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libm.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libfontconfig.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsndfile.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libFLAC.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libvorbis.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libvorbisenc.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libopus.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libogg.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmpg123.so* /usr/lib/x86_64-linux-gnu/
RUN true
COPY --from=builder /usr/lib/x86_64-linux-gnu/libmp3lame.so* /usr/lib/x86_64-linux-gnu/
RUN true

COPY . /app
ENV PYTHONPATH=/venv/lib/python3.11/site-packages
COPY --from=builder-venv /venv /venv
WORKDIR /app
CMD ["main.py"]
EXPOSE 7860