import os

import pytest

import app as app_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a Flask test client with an isolated temporary SQLite database."""
    test_db_path = os.path.join(tmp_path, "test_app.db")
    monkeypatch.setattr(app_module, "DATABASE_PATH", test_db_path)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False

    app_module.init_db()

    with app_module.app.test_client() as test_client:
        yield test_client
