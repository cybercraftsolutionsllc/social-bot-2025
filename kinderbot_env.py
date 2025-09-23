from pathlib import Path
import os, sys

# Load ~/.config/kinderbot/.env into env (no deps required).
DOTENV = Path.home() / ".config" / "kinderbot" / ".env"

def _manual_load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Try python-dotenv if installed; otherwise manual.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=DOTENV, override=True)
except Exception:
    _manual_load_env(DOTENV)

# Normalize key
if "OPENAI_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

# Optional sanity check with helpful hint (doesn't crash your run loop)
_key = os.getenv("OPENAI_API_KEY", "")
if not _key or not _key.startswith(("sk-", "sk-proj-")):
    sys.stderr.write(f"[kinderbot_env] WARN: OPENAI_API_KEY missing/malformed (looked in {DOTENV})\n")
