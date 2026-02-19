"""Adapter: cliente MCP que carga servidores externos y expone tools LangChain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from domain.models import MCPServerInfo, MCPStatus
from core.logger import logger


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "mcp_servers.json"


class MCPManager:
    """Gestiona conexiones a servidores MCP."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or CONFIG_PATH
        self._server_configs: List[Dict[str, Any]] = []
        self._sessions: List[ClientSession] = []
        self._exit_stack = AsyncExitStack()
        self._tools: List = []
        self._initialized = False
        self._load_config()

    def _load_config(self) -> None:
        if not self._config_path.exists():
            logger.warning(f"MCP config no encontrado: {self._config_path}")
            return
        try:
            with open(self._config_path) as f:
                data = json.load(f)
            for name, cfg in data.get("servers", {}).items():
                cfg["name"] = name
                if cfg.get("enabled", False):
                    self._server_configs.append(cfg)
                    logger.info(f"MCP server registrado: {name}")
        except Exception as e:
            logger.error(f"Error cargando config MCP: {e}")

    async def initialize(self) -> List:
        if self._initialized:
            return self._tools

        all_tools: List = []
        for cfg in self._server_configs:
            name = cfg["name"]
            try:
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env") or None,
                )
                transport = await self._exit_stack.enter_async_context(stdio_client(params))
                read_stream, write_stream = transport
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self._sessions.append(session)
                tools = await load_mcp_tools(session)
                all_tools.extend(tools)
                logger.info(f"MCP '{name}': {len(tools)} tools cargadas")
            except FileNotFoundError:
                logger.warning(f"MCP '{name}': comando '{cfg['command']}' no encontrado")
            except Exception as e:
                logger.warning(f"MCP '{name}' no conectó: {e}")

        self._tools = all_tools
        self._initialized = True
        logger.info(f"MCP: {len(all_tools)} tools de {len(self._sessions)} servidores")
        return all_tools

    async def cleanup(self) -> None:
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error cerrando MCP: {e}")
        self._sessions.clear()
        self._tools.clear()
        self._initialized = False

    @property
    def tools(self) -> List:
        return self._tools

    @property
    def server_count(self) -> int:
        return len(self._sessions)

    def get_status(self) -> MCPStatus:
        servers = [
            MCPServerInfo(
                name=cfg["name"],
                description=cfg.get("description", ""),
                enabled=cfg.get("enabled", False),
                command=cfg.get("command", ""),
            )
            for cfg in self._server_configs
        ]
        return MCPStatus(
            initialized=self._initialized,
            servers=servers,
            connected_servers=len(self._sessions),
            tool_count=len(self._tools),
            tool_names=[t.name for t in self._tools],
        )
