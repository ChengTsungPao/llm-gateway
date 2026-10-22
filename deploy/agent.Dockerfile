# Image for agent pods: the agent harness plus the tools agents usually need (Python, Node, git).
# Build from the repo root:  docker build -f deploy/agent.Dockerfile -t llm-gateway-agent:latest .
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm
# Node, npm and npx from the official Node image (same Debian base)
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system user.name "llm-agent" \
 && git config --system user.email "agent@localhost" \
 && git config --system --add safe.directory '*'

WORKDIR /opt/llm-gateway
COPY pyproject.toml README.md ./
COPY llm_gateway ./llm_gateway
RUN pip install --no-cache-dir . pytest

# /agent is the pod's own folder on the host: /agent/workspace (the agent's files) and /agent/state
WORKDIR /agent/workspace
EXPOSE 8000
CMD ["python", "-m", "llm_gateway.agent.podserver"]
