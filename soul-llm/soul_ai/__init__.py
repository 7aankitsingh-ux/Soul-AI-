"""
Soul AI Engine for SOUL-LLM
============================
Brings Soul AI's autonomous tools, knowledge base (RAG), system telemetry,
and ReAct agent reasoning loops directly into the SOUL-LLM ecosystem.

Usage in Python:
    import soul_ai
    from soul_ai import SoulAIEngine, execute_tool, get_system_info
"""

from .tools import (
    tools,
    execute_tool,
    get_tool_definitions,
    get_system_info,
    calculate,
    read_file,
    write_file,
    list_files
)
from .rag import SimpleRAG, rag_engine
from .agent import SoulAIEngine

__all__ = [
    "tools",
    "execute_tool",
    "get_tool_definitions",
    "get_system_info",
    "calculate",
    "read_file",
    "write_file",
    "list_files",
    "SimpleRAG",
    "rag_engine",
    "SoulAIEngine"
]
