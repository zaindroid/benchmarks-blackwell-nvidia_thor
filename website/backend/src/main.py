"""ThorBench leaderboard API — FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from api import router

app = FastAPI(title="ThorBench Leaderboard API", version="1.0.0")
app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
