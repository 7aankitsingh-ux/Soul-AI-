# start_server_fixed.py – Launch Soul‑AI FastAPI server and (optionally) expose it via ngrok

"""Utility script to run the FastAPI server (uvicorn) and optionally open a public
ngrok tunnel. This version ensures the ``api`` package can be imported by
adding the repository root to ``sys.path`` and runs the server from the repository
root directory.

Usage:
    python start_server_fixed.py

If ngrok fails (e.g., blocked by security software), the script will fall back
to a local-only endpoint at http://localhost:8000.
"""

import subprocess
import sys
import atexit
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Determine repository root (the folder containing ``api`` and ``src``)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
# Ensure the repo root is on sys.path so ``import api`` works
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Helper: clean up child processes on exit
# ---------------------------------------------------------------------------
_process: subprocess.Popen | None = None

def _cleanup() -> None:
    global _process
    if _process and _process.poll() is None:
        print("[INFO] Terminating FastAPI server process …")
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
    # Attempt to shut down any ngrok tunnel if it was started
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass

atexit.register(_cleanup)

# ---------------------------------------------------------------------------
# Start the FastAPI server (uvicorn) in a background process
# ---------------------------------------------------------------------------
def _start_fastapi() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    creation_flags = 0
    if sys.platform.startswith("win"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        text=True,
    )
    # Stream server logs to the console for visibility
    def _stream_output():
        for line in proc.stdout:  # type: ignore[union-attr]
            print(line, end="")
    import threading
    threading.Thread(target=_stream_output, daemon=True).start()
    return proc

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Launching Soul-AI FastAPI server ...")
    _process = _start_fastapi()

    # Give the server a moment to start before trying ngrok
    time.sleep(2)
    print("[INFO] Attempting to open ngrok tunnel …")
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8000, "http")
        print(f"\n[SUCCESS] Your API is publicly reachable at: {public_url}\n")
    except Exception as e:
        print(f"[WARN] ngrok tunnel could not be established: {e}")
        print("You can still access the API locally at http://localhost:8000")
    print("Press Ctrl+C to shut down the server.")

    try:
        while _process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Received interrupt – shutting down.")
        sys.exit(0)
