from __future__ import annotations
import datetime as dt, logging
from pathlib import Path
from .config import APP_NAME, APP_VERSION

def banner():
    print("\n" + "="*72); print(f"  {APP_NAME} v{APP_VERSION}"); print("  CIS*6520 — University of Guelph"); print("  Intended stakeholder context: RCMP NC3"); print("="*72)

def step(message: str):
    print("\n" + "="*72); print(f"  {message}"); print("="*72)

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def configure_logging(path: Path):
    ensure_parent(path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(path, encoding="utf-8")], force=True)
