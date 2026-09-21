from fastapi.testclient import TestClient

from api.main import app, service


def _fake_load() -> None:
	service.model = object()
	service.tokenizer = object()
	service.device = type("Device", (), {"type": "cpu"})()
	service.parameter_count = 25_523_712


def test_health() -> None:
	original_load = service.load
	service.load = _fake_load
	try:
		with TestClient(app) as client:
			response = client.get("/api/health")
		assert response.status_code == 200
		assert response.json() == {
			"status": "ok",
			"model": "Exp012",
			"parameters": 25_523_712,
			"device": "cpu",
		}
	finally:
		service.load = original_load


def test_empty_message_rejected() -> None:
	with TestClient(app, raise_server_exceptions=False) as client:
		response = client.post("/api/chat", json={"message": "   "})
	assert response.status_code == 422


def test_invalid_temperature_rejected() -> None:
	with TestClient(app, raise_server_exceptions=False) as client:
		response = client.post("/api/chat", json={"message": "Hello", "temperature": 0})
	assert response.status_code == 422


def test_invalid_top_p_rejected() -> None:
	with TestClient(app, raise_server_exceptions=False) as client:
		response = client.post("/api/chat", json={"message": "Hello", "top_p": 1.1})
	assert response.status_code == 422