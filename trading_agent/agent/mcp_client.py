from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from ..config import MCP_DIR
from .tools import Tool


def _safe_id(url: str) -> str:
    """Filesystem-safe identifier for a server URL."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", url).strip("_")[:80] or "mcp_server"


class FileTokenStorage(TokenStorage):
    """OAuth token + client info persisted as JSON, one file per server."""

    def __init__(self, server_url: str, base_dir: Path = MCP_DIR):
        base_dir.mkdir(parents=True, exist_ok=True)
        sid = _safe_id(server_url)
        self.tokens_path = base_dir / f"{sid}.tokens.json"
        self.client_path = base_dir / f"{sid}.client.json"

    async def get_tokens(self) -> OAuthToken | None:
        if not self.tokens_path.exists():
            return None
        return OAuthToken.model_validate_json(self.tokens_path.read_text(encoding="utf-8"))

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens_path.write_text(tokens.model_dump_json(), encoding="utf-8")

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        if not self.client_path.exists():
            return None
        return OAuthClientInformationFull.model_validate_json(
            self.client_path.read_text(encoding="utf-8")
        )

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_path.write_text(client_info.model_dump_json(), encoding="utf-8")


def _free_port() -> int:
    """Bind a random free port for the OAuth callback listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CallbackCatcher:
    """Tiny HTTP server that captures the OAuth redirect's `code` and `state`."""

    def __init__(self, port: int):
        self.port = port
        self.code: str | None = None
        self.state: str | None = None
        self._done = threading.Event()
        self._server: HTTPServer | None = None

    def start(self) -> str:
        parent = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                qs = parse_qs(urlparse(self.path).query)
                parent.code = (qs.get("code") or [None])[0]
                parent.state = (qs.get("state") or [None])[0]
                body = (
                    b"<html><body style='font-family:sans-serif;padding:32px'>"
                    b"<h2>Auth complete</h2>"
                    b"<p>You can close this tab and return to the terminal.</p>"
                    b"</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                parent._done.set()
            def log_message(self, *args, **kwargs):  # silence
                return

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        return f"http://127.0.0.1:{self.port}/callback"

    async def await_code(self, timeout: float = 300.0) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._done.wait(timeout=timeout))
        if self._server:
            self._server.shutdown()
        if not self.code:
            raise RuntimeError("OAuth callback timed out without a code")
        return self.code, self.state


def _build_oauth_provider(server_url: str) -> tuple[OAuthClientProvider, _CallbackCatcher]:
    catcher = _CallbackCatcher(_free_port())
    redirect_uri = catcher.start()
    metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        client_name="trading-agent",
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )

    async def redirect_handler(url: str) -> None:
        webbrowser.open(url)

    async def callback_handler() -> tuple[str, str | None]:
        return await catcher.await_code()

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(server_url),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    return provider, catcher


@dataclass
class RemoteMCPTool:
    name: str
    description: str
    input_schema: dict
    server_url: str

    def as_local_tool(self, server_url: str) -> Tool:
        """Wrap a remote tool so it slots into our existing Tool registry.

        The fn closes over the server URL and re-opens the MCP session per call.
        For high-throughput use we'd cache the session; for an agent loop with
        ~10 tool calls per session, per-call connection cost is acceptable.
        """
        async def _async_call(args: dict):
            provider, _ = _build_oauth_provider(server_url)
            async with streamablehttp_client(server_url, auth=provider) as (
                read, write, _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, args)
                    return [
                        c.text if hasattr(c, "text") else str(c) for c in result.content
                    ]

        def call(args: dict):
            args = {k: v for k, v in args.items() if not k.startswith("__")}
            return asyncio.run(_async_call(args))

        return Tool(
            name=f"mcp_{self.name}",
            description=f"[MCP: {server_url}] {self.description}",
            input_schema=self.input_schema,
            fn=call,
        )


async def _list_remote_tools(server_url: str) -> list[RemoteMCPTool]:
    provider, _ = _build_oauth_provider(server_url)
    async with streamablehttp_client(server_url, auth=provider) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            return [
                RemoteMCPTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                    server_url=server_url,
                )
                for t in listing.tools
            ]


def discover_tools(server_url: str) -> list[Tool]:
    """List the remote MCP server's tools and adapt them to our Tool dataclass."""
    remote_tools = asyncio.run(_list_remote_tools(server_url))
    return [rt.as_local_tool(server_url) for rt in remote_tools]


def is_authenticated(server_url: str) -> bool:
    storage = FileTokenStorage(server_url)
    return storage.tokens_path.exists() and storage.client_path.exists()
