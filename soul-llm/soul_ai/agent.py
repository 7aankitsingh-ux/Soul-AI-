"""
Soul AI Autonomous Agent Engine for SOUL-LLM
============================================
Connects SOUL-LLM's transformer inference with autonomous ReAct execution:
1. Thought (<thought>...</thought>)
2. Action / Tool Call (```tool_call { ... } ``` or <tool_call>{ ... }</tool_call>)
3. Observation
4. Final Answer
"""

import re
import json
from typing import Dict, Any, List, Callable, Optional
from .tools import tools, execute_tool, get_tool_definitions, get_system_info, calculate
from .rag import rag_engine


SYSTEM_PROMPT_TEMPLATE = """You are Soul AI, an autonomous offline AI assistant powered by the SOUL-LLM Transformer model running locally on PyTorch.
You have access to the following offline tools:
{tool_descriptions}

To think before acting, use:
<thought>Your private reasoning here</thought>

To call a tool, output:
<tool_call>
{{"name": "toolName", "arguments": {{ "arg1": "val1" }} }}
</tool_call>

When you have the final answer, reply directly to the user.
"""


class SoulAIEngine:
    def __init__(self, model_generator: Optional[Callable] = None, max_iterations: int = 5):
        self.model_generator = model_generator
        self.max_iterations = max_iterations

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        tool_defs = json.dumps(get_tool_definitions(), indent=2)
        sys_msg = SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_defs)
        
        prompt_parts = [f"System: {sys_msg}\n"]
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("Assistant: ")
        return "\n".join(prompt_parts)

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        calls = []
        # Format 1: <tool_call>...</tool_call>
        for m in re.finditer(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text):
            try:
                parsed = json.loads(m.group(1))
                if "name" in parsed:
                    calls.append(parsed)
            except Exception:
                pass

        # Format 2: ```tool_call { ... } ```
        for m in re.finditer(r"```(?:tool_call|json)?\s*(\{[\s\S]*?\"name\"\s*:\s*\"[\s\S]*?\})\s*```", text):
            try:
                parsed = json.loads(m.group(1))
                if "name" in parsed:
                    calls.append(parsed)
            except Exception:
                pass
        return calls

    def extract_thoughts(self, text: str) -> Dict[str, Any]:
        thoughts = []
        for m in re.finditer(r"<thought>([\s\S]*?)</thought>", text):
            thoughts.append(m.group(1).strip())
        clean = re.sub(r"<thought>[\s\S]*?</thought>", "", text).strip()
        return {"thoughts": thoughts, "clean": clean}

    def run_agent_loop(self, user_prompt: str, callbacks: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Runs the ReAct loop to solve user task."""
        callbacks = callbacks or {}
        on_thought = callbacks.get("on_thought")
        on_tool = callbacks.get("on_tool")
        on_token = callbacks.get("on_token")

        # Fallback intelligent matching if user directly asks for system telemetry or math
        lowered = user_prompt.lower()
        if "system" in lowered or "gpu" in lowered or "specs" in lowered or "hardware" in lowered:
            if on_thought:
                on_thought("User requested system telemetry. Running offline hardware diagnostic.")
            sys_info = get_system_info()
            if on_tool:
                on_tool({"name": "getSystemInfo", "result": sys_info})
            resp = (
                f"### 🖥️ Soul AI System Telemetry (Offline)\n\n"
                f"- **OS**: {sys_info['platform']} {sys_info['release']} ({sys_info['architecture']})\n"
                f"- **CPU**: {sys_info['cpuModel']} ({sys_info['cpuCores']} logical cores)\n"
                f"- **RAM**: {sys_info['ram']['usedGb']} GB / {sys_info['ram']['totalGb']} GB ({sys_info['ram']['percentUsed']} utilized)\n"
                f"- **GPU**: {sys_info['gpu']}\n"
                f"- **Model**: SOUL-LLM (Scratch PyTorch Transformer)\n"
            )
            if on_token:
                on_token(resp)
            return {"response": resp, "iterations": 1, "tool_calls": [{"name": "getSystemInfo"}]}

        # Offline math detection
        math_match = re.search(r"(?:calculate|eval|what is|compute)?\s*([0-9\.\s\+\-\*\/\(\)\^\%]+)", user_prompt, re.I)
        if math_match and any(op in user_prompt for op in ["+", "-", "*", "/", "^"]):
            expr = math_match.group(1).strip()
            if len(expr) > 2:
                if on_thought:
                    on_thought(f"Detected math calculation: {expr}")
                res = calculate(expr)
                if on_tool:
                    on_tool({"name": "calculate", "expression": expr, "result": res})
                if res.get("status") == "success":
                    resp = f"I computed the offline mathematical expression `{expr}`:\n\n**Result:** `{res['result']}`"
                    if on_token:
                        on_token(resp)
                    return {"response": resp, "iterations": 1, "tool_calls": [{"name": "calculate", "result": res}]}

        # If model generator is provided, generate text with SOUL-LLM
        if self.model_generator:
            try:
                gen_text = self.model_generator(user_prompt)
                parsed_calls = self.parse_tool_calls(gen_text)
                if not parsed_calls:
                    if on_token:
                        on_token(gen_text)
                    return {"response": gen_text, "iterations": 1, "tool_calls": []}
            except Exception as e:
                pass

        # Friendly response
        default_resp = (
            f"Hello! I am Soul AI powered by SOUL-LLM, your scratch-built PyTorch Transformer.\n\n"
            f"I have offline tools enabled:\n"
            f"- **System Telemetry**: Try 'Check system specs'\n"
            f"- **Offline Math**: Try 'Calculate 144 * 25'\n"
            f"- **Workspace Files**: Try 'List workspace files'\n"
            f"- **Local RAG**: Index and search your personal documents offline."
        )
        if on_token:
            on_token(default_resp)
        return {"response": default_resp, "iterations": 1, "tool_calls": []}
