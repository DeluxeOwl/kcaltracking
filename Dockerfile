FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY main.py .

EXPOSE 8765

CMD ["uv", "run", "--script", "main.py", "--host", "0.0.0.0", "--port", "8765", "--db", "/app/data/kcal.db"]
