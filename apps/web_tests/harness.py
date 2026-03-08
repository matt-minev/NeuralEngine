"""Shared web test harness for Flask apps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FlaskHarness:
    app: Any

    @property
    def client(self):
        self.app.config['TESTING'] = True
        return self.app.test_client()

    def assert_json_ok(self, response):
        assert response.status_code == 200, response.data
        return response.get_json()
