### Builder stage: install only CPU deps via uv
FROM ghcr.io/astral-sh/uv:debian-slim AS builder
WORKDIR /app

# Copy only lockfiles for caching
COPY pyproject.toml uv.lock ./

# Sync base dependencies (CPU-only) into .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Copy application source
COPY . /app

# Re-sync to install the package itself
RUN uv sync --locked


### Final stage: runtime image
FROM ghcr.io/astral-sh/uv:debian-slim
WORKDIR /app

# Copy pre-built venv from builder
COPY --from=builder /app/.venv /app/.venv

# Ensure venv bin is on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default command: run your agent script (adjust entrypoint as needed)
CMD ["uv", "run", "python", "agent.py"]
