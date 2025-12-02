FROM n8nio/n8n:latest

USER root

RUN apk update && apk add --no-cache \
    python3 \
    py3-pip \
    ffmpeg \
    py3-pillow \
    yt-dlp \
    wkhtmltopdf
USER node