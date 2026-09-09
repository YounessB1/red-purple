FROM node:22-slim AS node

FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive

# http.kali.org's redirector currently points package downloads (not the
# signed index) at a mirror whose TLS cert this base image's CA bundle
# won't validate. Package integrity is still enforced by APT's GPG-signed
# Release file + per-package hashes, independent of transport, so disabling
# peer verification here just unblocks the broken mirror.
RUN echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/99insecure-mirror-workaround

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

# Node.js runtime (copied from official image — no nodesource script needed)
# npm/npx are symlinks in the node image so we recreate them explicitly;
# COPY --from would dereference the symlink and break relative require() paths.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# OpenCode CLI
RUN npm install -g opencode-ai@1.14.50

# Python dependencies
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir litellm fastapi "uvicorn[standard]"

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

CMD ["python3", "-m", "uvicorn", "source.agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
