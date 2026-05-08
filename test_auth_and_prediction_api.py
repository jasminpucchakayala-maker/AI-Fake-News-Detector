import app as app_module


def register_user(client, name="Test User", email="test@example.com", password="password123"):
    return client.post(
        "/register",
        data={"name": name, "email": email, "password": password},
        follow_redirects=True,
    )


def login_user(client, email="test@example.com", password="password123"):
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=True,
    )


def test_user_registration_and_login_flow(client):
    response_register = register_user(client)
    assert response_register.status_code == 200
    assert b"Registration successful" in response_register.data

    response_login = login_user(client)
    assert response_login.status_code == 200
    assert b"Welcome back!" in response_login.data


def test_login_rejects_invalid_credentials(client):
    register_user(client)
    response = client.post(
        "/login",
        data={"email": "test@example.com", "password": "wrong-password"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_api_predict_returns_prediction_payload(client, monkeypatch):
    monkeypatch.setattr(app_module, "predict_news", lambda _text: ("FAKE", 88.5))
    monkeypatch.setattr(app_module, "evaluate_source_credibility", lambda _url: ("TRUSTED", 90))

    response = client.post(
        "/api/predict",
        json={
            "article_text": "This is a long enough article text with more than thirty characters for prediction.",
            "source_url": "https://reuters.com/sample-news",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "FAKE"
    assert payload["confidence"] == 88.5
    assert payload["source_label"] == "TRUSTED"
    assert payload["source_score"] == 90


def test_api_predict_validates_short_text(client):
    response = client.post("/api/predict", json={"article_text": "too short"})
    assert response.status_code == 400
    payload = response.get_json()
    assert "article_text must contain at least 30 characters" in payload["error"]
