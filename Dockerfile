# Set default values for build arguments
ARG PARENT_VERSION=latest-3.13
ARG PORT=8085
ARG PORT_DEBUG=8086

FROM defradigital/python-development:${PARENT_VERSION} AS development

USER root
RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y gnupg curl \
    && curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-8.0.gpg \
    && echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/debian bookworm/mongodb-org/8.0 main" > /etc/apt/sources.list.d/mongodb-org-8.0.list \
    && apt update \
    && apt install -y mongodb-mongosh \
    && rm -rf /var/lib/apt/lists/*
USER nonroot

ENV PATH="/home/nonroot/.venv/bin:/home/nonroot/.local/bin:${PATH}"
ENV LOG_CONFIG="logging-dev.json"


WORKDIR /home/nonroot

COPY --chown=nonroot:nonroot pyproject.toml .
COPY --chown=nonroot:nonroot README.md .
COPY --chown=nonroot:nonroot uv.lock .
COPY --chown=nonroot:nonroot app/ ./app/
COPY --chown=nonroot:nonroot entrypoint.sh .
COPY --chown=nonroot:nonroot perf-tests/ ./perf-tests/

#Canned data for perf-tests
RUN chmod +x entrypoint.sh  \
    ./perf-tests/scripts/init-mongodb.sh

RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --link-mode=copy

COPY --chown=nonroot:nonroot logging-dev.json .

ARG PORT=8085
ARG PORT_DEBUG=8086
ENV PORT=${PORT}
EXPOSE ${PORT} ${PORT_DEBUG}

CMD [ "/home/nonroot/.venv/bin/ai-defra-search-agent" ]

FROM defradigital/python:${PARENT_VERSION} AS production

ENV PATH="/home/nonroot/.venv/bin:${PATH}"
ENV LOG_CONFIG="logging.json"

USER root

RUN apt update && \
    apt install -y curl

USER nonroot

WORKDIR /home/nonroot

COPY --from=development /home/nonroot/pyproject.toml .
COPY --chown=nonroot:nonroot README.md .
COPY --from=development /home/nonroot/uv.lock .
COPY --from=development /home/nonroot/app ./app

RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=1000,gid=1000 \
    --mount=from=development,source=/home/nonroot/.local/bin/uv,target=/home/nonroot/.local/bin/uv \
    uv sync --locked --compile-bytecode --link-mode=copy --no-dev

COPY logging.json .

ARG PORT
ENV PORT=${PORT}
EXPOSE ${PORT}

ENTRYPOINT [ "./entrypoint.sh" ]
