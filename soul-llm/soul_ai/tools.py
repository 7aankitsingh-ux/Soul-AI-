"""
Soul AI Tool Suite for SOUL-LLM
===============================
Provides offline tools for system telemetry, math calculations,
workspace file operations, and knowledge retrieval.
"""

import os
import platform
import math
import subprocess
import ctypes
from typing import Dict, Any, List

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def get_system_info() -> Dict[str, Any]:
    """Inspect local hardware, CPU, RAM, and GPU using standard library."""
    total_gb = 16.0
    used_gb = 8.0
    available_gb = 8.0
    percent_used = "50%"

    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
            available_gb = round(stat.ullAvailPhys / (1024 ** 3), 1)
            used_gb = round((stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3), 1)
            percent_used = f"{stat.dwMemoryLoad}%"
    except Exception:
        pass

    gpu_info = "Standard System Display"
    try:
        wmic_output = subprocess.check_output(
            ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True,
            timeout=3
        )
        lines = [line.strip() for line in wmic_output.strip().split("\n") if line.strip()]
        if lines:
            gpu_info = ", ".join(lines)
    except Exception:
        pass

    return {
        "platform": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpuModel": platform.processor(),
        "cpuCores": os.cpu_count() or 1,
        "ram": {
            "totalGb": total_gb,
            "usedGb": used_gb,
            "freeGb": available_gb,
            "percentUsed": percent_used
        },
        "gpu": gpu_info,
        "workspacePath": WORKSPACE_DIR
    }


def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluates a basic mathematical expression."""
    allowed_names = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "pi": math.pi, "e": math.e, "pow": math.pow, "abs": abs,
        "round": round
    }
    cleaned = expression.replace("^", "**")
    try:
        result = eval(cleaned, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result, "status": "success"}
    except Exception as e:
        return {"expression": expression, "error": str(e), "status": "error"}


def read_file(path: str) -> Dict[str, Any]:
    """Reads content of a file within the workspace."""
    target_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not target_path.startswith(WORKSPACE_DIR):
        return {"error": "Access denied: Path is outside workspace.", "status": "error"}
    if not os.path.exists(target_path):
        return {"error": f"File '{path}' not found.", "status": "error"}

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"path": path, "content": content, "sizeBytes": len(content), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Writes content to a file within the workspace."""
    target_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not target_path.startswith(WORKSPACE_DIR):
        return {"error": "Access denied: Path is outside workspace.", "status": "error"}

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path, "bytesWritten": len(content), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


def list_files(dir_path: str = ".") -> Dict[str, Any]:
    """Lists files within the workspace directory."""
    target_dir = os.path.abspath(os.path.join(WORKSPACE_DIR, dir_path))
    if not target_dir.startswith(WORKSPACE_DIR):
        return {"error": "Access denied: Path is outside workspace.", "status": "error"}
    if not os.path.exists(target_dir):
        return {"error": f"Directory '{dir_path}' does not exist.", "status": "error"}

    entries = []
    try:
        for entry in os.scandir(target_dir):
            entries.append({
                "name": entry.name,
                "isDirectory": entry.is_dir(),
                "sizeBytes": entry.stat().st_size if entry.is_file() else 0
            })
        return {"directory": dir_path, "files": entries, "count": len(entries), "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


tools = {
    "getSystemInfo": {
        "name": "getSystemInfo",
        "description": "Inspect offline system hardware, CPU, RAM, and GPU specifications.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "handler": lambda args: get_system_info()
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate mathematical expressions (e.g., '144 * 25', 'sqrt(256)').",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate"}
            },
            "required": ["expression"]
        },
        "handler": lambda args: calculate(args.get("expression", ""))
    },
    "readFile": {
        "name": "readFile",
        "description": "Read text content from a file inside the local workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file in workspace"}
            },
            "required": ["path"]
        },
        "handler": lambda args: read_file(args.get("path", ""))
    },
    "writeFile": {
        "name": "writeFile",
        "description": "Write or overwrite text content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file in workspace"},
                "content": {"type": "string", "description": "Text content to write"}
            },
            "required": ["path", "content"]
        },
        "handler": lambda args: write_file(args.get("path", ""), args.get("content", ""))
    },
    "listFiles": {
        "name": "listFiles",
        "description": "List files and subdirectories within the local workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Relative path in workspace (default: '.')"}
            },
            "required": []
        },
        "handler": lambda args: list_files(args.get("dir_path", "."))
    }
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Returns tool schemas formatted for LLM prompts."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
        for t in tools.values()
    ]


def execute_tool(name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a tool by name with arguments."""
    args = args or {}
    if name not in tools:
        return {"error": f"Tool '{name}' not found. Available: {list(tools.keys())}", "status": "error"}
    try:
        return tools[name]["handler"](args)
    except Exception as e:
        return {"error": str(e), "status": "error"}
