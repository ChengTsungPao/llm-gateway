FROM python:3.12-slim-bookworm

# System dependencies: curl for healthcheck/api probes, git and kubectl for K8s agent pod host
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
 && curl -fsSL -o /usr/local/bin/kubectl https://dl.k8s.io/release/v1.29.2/bin/linux/amd64/kubectl \
 && chmod +x /usr/local/bin/kubectl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies and llm-gateway package
COPY pyproject.toml README.md ./
COPY llm_gateway ./llm_gateway
RUN pip install --no-cache-dir .

ENV GATEWAY_HOST=0.0.0.0
ENV PORT=8790
EXPOSE 8790

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8790/api/health || exit 1

CMD ["python", "-m", "llm_gateway", "--config", "config.json"]
