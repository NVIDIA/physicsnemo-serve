# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight HTTP health stub for Lepton.AI startup probe.

Listens on port 8001 and proxies all requests to the real inference server on
port 8080. If the backend isn't ready yet, /health returns 200 so Lepton's
health probe stays satisfied while the service boots.

Based on script provided by Lepton.AI support.
"""

import http.server
import socketserver
import urllib.error
import urllib.request


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _proxy(self, method):
        body = (
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if method == "POST"
            else None
        )
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:8080{self.path}",
                method=method,
                data=body,
                headers={k: v for k, v in self.headers.items() if k.lower() != "host"},
            )
            r = urllib.request.urlopen(req, timeout=600)
            self.send_response(r.status)
            for k, v in r.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(r.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception:
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"loading"}')
            else:
                self.send_error(503, "loading")

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")


socketserver.ThreadingTCPServer.allow_reuse_address = True
socketserver.ThreadingTCPServer(("0.0.0.0", 8001), H).serve_forever()
