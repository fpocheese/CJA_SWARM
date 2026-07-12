#!/usr/bin/env python3
"""Small JSON-lines protocol helpers for V71 HIL split simulation."""

from __future__ import annotations

import json
import socket
from typing import Any


class JsonLineSocket:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb", buffering=0)

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.file.write(payload + b"\n")

    def recv(self) -> dict[str, Any]:
        line = self.file.readline()
        if not line:
            raise ConnectionError("peer closed connection")
        data = json.loads(line.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object, got {type(data)!r}")
        return data

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            self.sock.close()


def listen(host: str, port: int, backlog: int = 16) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(port)))
    sock.listen(backlog)
    return sock


def connect(host: str, port: int, timeout_s: float = 30.0) -> JsonLineSocket:
    sock = socket.create_connection((host, int(port)), timeout=timeout_s)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return JsonLineSocket(sock)
