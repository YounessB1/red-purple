FROM kalilinux/kali-rolling

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install litellm fastapi "uvicorn[standard]"

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

CMD ["python3", "-m", "uvicorn", "source.server:app", "--host", "0.0.0.0", "--port", "8000"]
