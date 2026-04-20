FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive

# Core system + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    curl wget git sudo ca-certificates \
    # Network utilities
    net-tools dnsutils whois tcpdump socat \
    # Reconnaissance
    nmap nikto gobuster dirb \
    # Web exploitation
    sqlmap wfuzz \
    # Credential attacks
    hydra john sshpass \
    # Network tools
    netcat-openbsd proxychains4 \
    # Text processing
    jq ripgrep \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install ffuf (architecture-aware)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then FFUF_ARCH="linux_amd64"; \
    elif [ "$ARCH" = "arm64" ]; then FFUF_ARCH="linux_arm64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    wget -q "https://github.com/ffuf/ffuf/releases/download/v2.1.0/ffuf_2.1.0_${FFUF_ARCH}.tar.gz" -O /tmp/ffuf.tar.gz && \
    tar -xzf /tmp/ffuf.tar.gz -C /tmp && \
    mv /tmp/ffuf /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffuf && \
    rm /tmp/ffuf.tar.gz

# Download common wordlists from SecLists
RUN mkdir -p /usr/share/wordlists && \
    (curl -fsSL --retry 3 --retry-delay 5 \
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt" \
        -o /usr/share/wordlists/common.txt || true) && \
    (curl -fsSL --retry 3 --retry-delay 5 \
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/raft-medium-directories.txt" \
        -o /usr/share/wordlists/raft-medium-directories.txt || true) && \
    (curl -fsSL --retry 3 --retry-delay 5 \
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt" \
        -o /usr/share/wordlists/top-10000-passwords.txt || true)

# Install linpeas for privilege escalation enumeration
RUN mkdir -p /opt/peass && \
    wget -q https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh -O /opt/peass/linpeas.sh && \
    chmod +x /opt/peass/linpeas.sh

# Python dependencies
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir litellm fastapi "uvicorn[standard]"

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

CMD ["python3", "-m", "uvicorn", "source.agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
