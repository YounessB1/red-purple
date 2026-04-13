FROM kalilinux/kali-rolling

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install "litellm==1.81.1" --break-system-packages

WORKDIR /app

CMD ["python3", "-m", "redpurple.agent"]
