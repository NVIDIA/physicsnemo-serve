# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal GitHub REST client (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://api.github.com"
PAGE_SIZE = 100
MAX_PAGES = 10


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GitHub:
    def __init__(self, repo: str, token: str, api: str = API):
        self.repo = repo
        self._token = token
        self._api = api.rstrip("/")
        self._no_redirect = urllib.request.build_opener(_NoRedirect)

    def _url(self, path: str, params: dict | None = None) -> str:
        path = path.format(repo=self.repo)
        url = path if path.startswith("http") else f"{self._api}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def _request(
        self, method: str, url: str, body: Any = None
    ) -> urllib.request.Request:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return request

    def request(
        self, method: str, path: str, body: Any = None, params: dict | None = None
    ) -> Any:
        with urllib.request.urlopen(
            self._request(method, self._url(path, params), body), timeout=60
        ) as response:
            payload = response.read()
        return json.loads(payload) if payload else None

    def get(self, path: str, params: dict | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: Any) -> Any:
        return self.request("POST", path, body=body)

    def patch(self, path: str, body: Any) -> Any:
        return self.request("PATCH", path, body=body)

    def paginate(
        self, path: str, key: str | None = None, params: dict | None = None
    ) -> list:
        items: list = []
        for page in range(1, MAX_PAGES + 1):
            data = self.get(
                path, {**(params or {}), "per_page": PAGE_SIZE, "page": page}
            )
            batch = data[key] if key else data
            items.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
        return items

    def get_text(self, path: str) -> str:
        """GET an endpoint that redirects to a pre-signed download (e.g. job logs).

        The redirect target is fetched without the Authorization header, which
        blob storage rejects.
        """
        try:
            with self._no_redirect.open(
                self._request("GET", self._url(path)), timeout=60
            ) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers["Location"]
        with urllib.request.urlopen(location, timeout=120) as response:
            return response.read().decode("utf-8", errors="replace")
