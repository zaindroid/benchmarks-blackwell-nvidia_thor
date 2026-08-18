"""Tests for the community submission portal API."""

from fastapi.testclient import TestClient

from api import _SUBMISSIONS, router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)


def _reset():
    _SUBMISSIONS.clear()


def test_submit_model_pending():
    _reset()
    with TestClient(app) as client:
        response = client.post("/api/submissions", json={
            "model_id": "community/fast-detector",
            "architecture": "cnn",
            "parameters": 5000000,
            "contact_email": "researcher@example.com",
            "metrics": {"latency_p50_ms": 2.4, "throughput": 420.0},
        })
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        sub = body["submission"]
        assert sub["submission_id"].startswith("sub-")
        assert sub["status"] == "pending"
        assert sub["metrics"]["latency_p50_ms"] == 2.4


def test_list_and_review_submission():
    _reset()
    with TestClient(app) as client:
        created = client.post("/api/submissions", json={
            "model_id": "community/fast-detector",
            "metrics": {"latency_p50_ms": 2.4},
        }).json()["submission"]
        sub_id = created["submission_id"]

        pending = client.get("/api/submissions", params={"status": "pending"}).json()
        assert pending["count"] == 1

        reviewed = client.post(f"/api/submissions/{sub_id}/review", json={
            "status": "approved", "comment": "verified on reference node",
        })
        assert reviewed.status_code == 200
        assert reviewed.json()["submission"]["status"] == "approved"

        approved = client.get("/api/submissions", params={"status": "approved"}).json()
        assert approved["count"] == 1
        assert approved["submissions"][0]["review_comment"] == "verified on reference node"


def test_review_unknown_submission_404():
    with TestClient(app) as client:
        response = client.post("/api/submissions/sub-9999/review", json={"status": "approved"})
        assert response.status_code == 404


def test_invalid_review_status_rejected():
    with TestClient(app) as client:
        created = client.post("/api/submissions", json={"model_id": "m"}).json()["submission"]
        response = client.post(
            f"/api/submissions/{created['submission_id']}/review",
            json={"status": "maybe"},
        )
        assert response.status_code == 422
