# Need C module to be compiled
FROM debian:12-slim AS builder
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes \
    python3-venv gcc libpython3-dev libjpeg-dev zlib1g-dev libfreetype6-dev fontconfig && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Install Google Fonts
# This will need to be slimmified at some point as this adds between 1-2GB. Luckily, since this is a multi-stage build
# we can just cherry-pick the fonts we want but it would be nice to find out how to get specific fonts and directly load
# them in the final step/build.
ADD https://github.com/google/fonts/archive/main.tar.gz /tmp/gfonts.tar.gz
RUN tar -xf /tmp/gfonts.tar.gz && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find $PWD/fonts-main/ -name "*.ttf" -exec install -m644 {} /usr/share/fonts/truetype/google-fonts/ \; && \
    fc-cache -f

# Install dependencies. This is done in a separate step to take advantage of Docker's caching, so that we don't have to
# reinstall dependencies every time we change the code.
FROM builder as builder-venv
COPY requirements.txt requirements.txt
RUN /venv/bin/pip install --disable-pip-version-check -r requirements.txt

FROM gcr.io/distroless/python3-debian12
# Copy over dependencies needed needed for Pillow to work.
COPY --from=builder /usr/lib/aarch64-linux-gnu/libjpeg.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libz.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libfreetype.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libz.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libpng16.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libbrotlidec.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libbrotlicommon.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libm.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libfontconfig.so* /usr/lib/aarch64-linux-gnu/

# Copy fonts to the final image, so it does not take too much space.
# Display
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Roboto* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/OpenSans* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Montserrat* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Lato* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Poppins* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Oswald* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Slabo* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/SpaceMono* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/FiraSans* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Rosario* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Ubuntu* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Cormorant* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Aboreto* /usr/share/fonts/truetype/google-fonts/
# Print
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Merriweather* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/EBGaramond* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/PlayfairDisplay* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Alegreya* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Lora* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Neuton* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Spectral* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Rosarivo* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Vollkorn* /usr/share/fonts/truetype/google-fonts/
# Handwriting
COPY --from=builder /usr/share/fonts/truetype/google-fonts/DancingScript* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/IndieFlower* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/HomemadeApple* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/PatrickHand* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/AmaticSC* /usr/share/fonts/truetype/google-fonts/
# Misc
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Rubik* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/PressStart2P* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/Silkscreen* /usr/share/fonts/truetype/google-fonts/
COPY --from=builder /usr/share/fonts/truetype/google-fonts/PirataOne* /usr/share/fonts/truetype/google-fonts/

COPY . /app
ENV PYTHONPATH=/venv/lib/python3.11/site-packages
COPY --from=builder-venv /venv /venv
WORKDIR /app
CMD ["main.py"]
EXPOSE 7860