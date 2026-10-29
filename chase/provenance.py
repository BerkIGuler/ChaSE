"""The provenance that every result file records. That is the version of ChaSE, its git commit and checksums."""

import hashlib
import subprocess
from pathlib import Path

import chase

REPO_DIR = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    out = subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def code_state() -> dict:
    """The version of ChaSE and, for a git checkout of ChaSE, its commit and whether the working tree differs from
    it. An installed copy inside another repository records no commit."""
    top = _git("rev-parse", "--show-toplevel")
    ours = top is not None and Path(top).resolve() == REPO_DIR
    state = {"version": chase.__version__, "commit": _git("rev-parse", "HEAD") if ours else None}
    if state["commit"] is not None:
        state["dirty"] = bool(_git("status", "--porcelain", "--", "chase", "scripts"))
    return state
