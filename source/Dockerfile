FROM kalilinux/kali-rolling

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install "litellm==1.81.1" langfuse --break-system-packages

WORKDIR /app

CMD ["python3", "-m", "source.agent"]
