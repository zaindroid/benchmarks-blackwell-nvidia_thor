# Leaderboard API image.
FROM python:3.12-slim

WORKDIR /app
COPY website/backend /app
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["thor-leaderboard"]
