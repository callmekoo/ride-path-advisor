# One image for the server and for the batch commands (rpa update, rpa sync-tracks, …).
# Named Dockerfile so that both docker and podman find it without being told.
#
#   podman build -t rpa .
#   podman run -d --name rpa --userns=keep-id -v ./data:/data:Z -p 127.0.0.1:8000:8000 --env-file .env rpa
#   podman run --rm --userns=keep-id -v ./data:/data:Z rpa update
#
#   docker compose up -d --build        (see compose.yaml)
#   docker compose run --rm rpa update

# --- Python environment with the Rust kernel compiled into it
FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Third-party packages first: this layer survives changes to our own code.
COPY pyproject.toml uv.lock README.md ./
COPY kernel/pyproject.toml kernel/pyproject.toml
RUN uv sync --frozen --no-dev --no-install-workspace
COPY kernel kernel
COPY src src
RUN uv sync --frozen --no-dev --no-editable

# --- The web page
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

# --- What runs
FROM python:3.14-slim
# osmium-tool merges and clips the OSM extracts in `rpa fetch-region` and `rpa update`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends osmium-tool \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/.venv /app/.venv
COPY --from=web /web/dist /app/web
# HOME: the process runs as the owner of the data directory, a user that has no home in the image.
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    RPA_DATA_DIR=/data \
    RPA_WEB_DIR=/app/web
WORKDIR /app
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["rpa"]
# No logins: publish the port on a trusted interface only (see README).
CMD ["serve", "--host", "0.0.0.0"]
