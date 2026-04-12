"""
start.py  Cross-platform launcher for Stock Platform services.
Usage:
    python start.py           - start all services
    python start.py stop      - stop all services
    python start.py status    - check status
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
LOG_DIR  = ROOT / "logs"
PID_FILE = ROOT / ".service_pids.json"

STOCKENV_PY = ROOT / "stockenv" / "Scripts" / "python.exe"
PYTHON = str(STOCKENV_PY) if STOCKENV_PY.exists() else sys.executable

SERVICES = [
    {"name": "ingestion",   "script": "run_ingestion.py",   "port": 8001},
    {"name": "inference",   "script": "run_inference.py",   "port": 8002},
    {"name": "llm_insight", "script": "run_llm_insight.py", "port": 8003},
    {"name": "agent",       "script": "run_agent.py",       "port": 8004},
    {"name": "gateway",     "script": "run_gateway.py",     "port": 8000},
]

G = "\033[92m"
R = "\033[91m"
C = "\033[96m"
Y = "\033[93m"
X = "\033[0m"


def ok(msg):
    print(f"  {G}[OK]{X}  {msg}")


def err(msg):
    print(f"  {R}[ERR]{X} {msg}")


def info(msg):
    print(f"  {C}[..]{X}  {msg}")


def hdr(msg):
    print(f"\n{Y}{msg}{X}")


def load_pids():
    if PID_FILE.exists():
        return json.loads(PID_FILE.read_text())
    return {}


def save_pids(pids):
    PID_FILE.write_text(json.dumps(pids, indent=2))


def kill_port(port):
    """Kill whatever process is holding a port."""
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                f"netstat -ano | findstr :{port}",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            seen = set()
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pid = int(parts[-1])
                    if pid > 0 and pid not in seen:
                        seen.add(pid)
                        subprocess.call(
                            f"taskkill /PID {pid} /F",
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        info(f"Freed port {port} (PID {pid})")
        except Exception:
            pass
    else:
        try:
            subprocess.call(
                f"fuser -k {port}/tcp",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def is_up(port, timeout=3):
    try:
        urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=timeout
        )
        return True
    except Exception:
        return False


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def stop():
    hdr("Stopping services...")
    pids = load_pids()
    for svc in SERVICES:
        name = svc["name"]
        pid  = pids.get(name)
        if pid:
            try:
                if sys.platform == "win32":
                    subprocess.call(
                        f"taskkill /PID {pid} /F /T",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    os.kill(pid, signal.SIGTERM)
                ok(f"{name} (PID {pid}) stopped")
            except Exception:
                info(f"{name} already gone")
    for svc in SERVICES:
        kill_port(svc["port"])
    if PID_FILE.exists():
        PID_FILE.unlink()
    print()


def status():
    hdr("Service Status")
    pids = load_pids()
    for svc in SERVICES:
        name = svc["name"]
        port = svc["port"]
        pid  = pids.get(name)
        alive = False
        if pid:
            try:
                if sys.platform == "win32":
                    out = subprocess.check_output(
                        f"tasklist /FI \"PID eq {pid}\"",
                        shell=True,
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    alive = str(pid) in out
                else:
                    os.kill(pid, 0)
                    alive = True
            except Exception:
                pass
        up = is_up(port)
        if up:
            ok(f"{name}  :{port}  HTTP OK")
        elif alive:
            info(f"{name}  :{port}  running but not responding yet")
        else:
            err(f"{name}  :{port}  not running")
    print()


def start():
    load_env()

    print()
    print(f"  {C}====================================================={X}")
    print(f"  {C}      Stock Platform  -  Python Launcher{X}")
    print(f"  {C}  Ingestion  Inference  Grok Insight  Gateway{X}")
    print(f"  {C}====================================================={X}")
    print()

    ok(f"Python: {PYTHON}")
    LOG_DIR.mkdir(exist_ok=True)

    hdr("Clearing ports 8000-8003...")
    for svc in SERVICES:
        kill_port(svc["port"])
    time.sleep(2)

    xai_key = os.environ.get("XAI_API_KEY", "")
    if not xai_key:
        print()
        print(f"  {Y}Enter your xAI Grok API key (press Enter to skip):{X}")
        key = input("  XAI_API_KEY: ").strip()
        if key:
            os.environ["XAI_API_KEY"] = key
            env_file = ROOT / ".env"
            lines = []
            if env_file.exists():
                lines = [
                    ln for ln in env_file.read_text().splitlines()
                    if not ln.startswith("XAI_API_KEY")
                ]
            lines.append(f"XAI_API_KEY={key}")
            env_file.write_text("\n".join(lines))
            ok("XAI_API_KEY saved to .env")
        else:
            info("Skipping - rule-based fallback will be used")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Re-read .env to make sure all keys are in subprocess env
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    xai = env.get("XAI_API_KEY", "")
    if xai:
        ok(f"XAI_API_KEY: {xai[:8]}...{xai[-4:]} (loaded)")
    else:
        info("XAI_API_KEY not set - LLM will use rule-based fallback")

    hdr("Starting services...")
    pids = {}

    for svc in SERVICES:
        name   = svc["name"]
        script = str(ROOT / svc["script"])
        port   = svc["port"]

        log_out = open(LOG_DIR / f"{name}.log",     "w")
        log_err = open(LOG_DIR / f"{name}.err.log", "w")

        proc = subprocess.Popen(
            [PYTHON, script],
            cwd=str(ROOT),
            env=env,
            stdout=log_out,
            stderr=log_err,
        )
        pids[name] = proc.pid
        ok(f"{name} started  PID={proc.pid}")
        time.sleep(4)

    save_pids(pids)

    hdr("Waiting for health checks...")
    all_up = True

    for svc in SERVICES:
        name = svc["name"]
        port = svc["port"]
        up   = False
        info(f"Checking {name} on :{port} ...")
        for _ in range(30):
            if is_up(port):
                up = True
                break
            time.sleep(2)
        if up:
            ok(f"{name}  http://localhost:{port}  UP")
        else:
            err(f"{name}  http://localhost:{port}  FAILED")
            log_path = LOG_DIR / f"{name}.err.log"
            if log_path.exists():
                lines = log_path.read_text().strip().splitlines()
                for line in lines[-8:]:
                    print(f"    {R}{line}{X}")
            all_up = False

    print()
    if all_up:
        print(f"  {G}====================================================={X}")
        print(f"  {G}        All services are running!{X}")
        print()
        print(f"  {G}  Gateway API   http://localhost:8000{X}")
        print(f"  {G}  Swagger docs  http://localhost:8000/docs{X}")
        print(f"  {G}  Dashboard     open dashboard.html in browser{X}")
        print()
        print(f"  {G}  Stop:    python start.py stop{X}")
        print(f"  {G}  Status:  python start.py status{X}")
        print(f"  {G}====================================================={X}")

        dash = ROOT / "dashboard.html"
        if dash.exists():
            time.sleep(2)
            info("Opening dashboard...")
            if sys.platform == "win32":
                os.startfile(str(dash))
            else:
                subprocess.call(["xdg-open", str(dash)])
    else:
        print(f"  {R}Some services failed - check the logs/ folder{X}")

    print()


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "start"
    if cmd == "stop":
        stop()
    elif cmd == "status":
        status()
    else:
        start()