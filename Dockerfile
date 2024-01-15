# Need C module to be compiled
FROM debian:12-slim AS builder
ARG TARGETARCH
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes \
    python3-venv gcc libpython3-dev libjpeg-dev zlib1g-dev libfreetype6-dev libsndfile1-dev fontconfig ffmpeg && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Copy files to a unified location to call in the next stage, regardless of architecture.
RUN mkdir -p /usr/lib/unified/
RUN if [ "$TARGETARCH" = "arm64" ] ; then \
      ln -s /usr/lib/aarch64-linux-gnu/lib*.so* /usr/lib/unified/; \
    elif [ "$TARGETARCH" = "amd64" ] ; then \
      ln -s /usr/lib/x86_64-linux-gnu/lib*.so* /usr/lib/unified/; \
    fi

# Copy ffmpeg and its dependencies.
RUN mkdir -p /usr/lib/unified/ffmpeg-deps
RUN cp $(which ffmpeg) $(which ffprobe) /usr/lib/unified/ffmpeg-deps
RUN ldd $(which ffmpeg) | tr -s '[:space:]' '\n' | grep '^/' | \
    xargs -I % sh -c 'cp % /usr/lib/unified/ffmpeg-deps' \
RUN ldd $(which ffprobe) | tr -s '[:space:]' '\n' | grep '^/' | \
    xargs -I % sh -c 'cp % /usr/lib/unified/ffmpeg-deps'

# Install dependencies. This is done in a separate step to take advantage of Docker's caching, so that we don't have to
# reinstall dependencies every time we change the code.
FROM builder as builder-venv
COPY requirements.txt requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

FROM gcr.io/distroless/python3-debian12
# Copy over dependencies needed needed for Pillow to work.
COPY --from=builder /usr/lib/unified/ffmpeg-deps /usr/local/bin/
COPY --from=builder /usr/lib/unified/libjpeg.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libz.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libfreetype.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libz.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libpng16.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libbrotlidec.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libbrotlicommon.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libm.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libfontconfig.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libsndfile.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libFLAC.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libvorbis.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libvorbisenc.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libopus.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libogg.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libmpg123.so* /usr/lib/
COPY --from=builder /usr/lib/unified/libmp3lame.so* /usr/lib/

COPY . /app
ENV PYTHONPATH=/venv/lib/python3.11/site-packages
ENV LD_LIBRARY_PATH=/usr/local/bin
COPY --from=builder-venv /venv /venv
WORKDIR /app
CMD ["main.py"]
EXPOSE 7860