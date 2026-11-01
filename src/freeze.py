"""
Code freeze record: SHA-256 of every strategy, contrib and harness file plus the dependency lock and
the Dockerfile, in `sha256sum` format, so the official run can prove it used the frozen code.

    python -m src.freeze write [--image oqs-sandbox]   # writes benchmarks/freeze.txt
    python -m src.freeze check                         # exit 1 if any file changed, is missing or new
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FREEZE_FILE = Path("benchmarks/freeze.txt")
PATTERNS = ("strategies/*.py", "src/**/*.py", "requirements.lock", "Dockerfile")


def frozen_files(root: Path) -> list[str]:
    files = {p.relative_to(root).as_posix() for pat in PATTERNS for p in root.glob(pat) if p.is_file()}
    return sorted(f for f in files if "__pycache__" not in f)


def file_hashes(root: Path) -> dict[str, str]:
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in frozen_files(root)}


def _run(cmd: list[str], root: Path) -> str:
    try:
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "n/a"


def write(root: Path, out: Path, image: str = "") -> None:
    header = [f"# commit {_run(['git', 'rev-parse', 'HEAD'], root)}"]
    if image:
        header.append(f"# image {image} {_run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], root)}")
    body = [f"{h}  {f}" for f, h in file_hashes(root).items()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + body) + "\n", encoding="utf-8", newline="\n")


def check(root: Path, freeze: Path) -> list[str]:
    """Readable problems; an empty list means the tree matches the freeze record."""
    recorded = {}
    for line in freeze.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            h, f = line.split("  ", 1)
            recorded[f] = h
    current = file_hashes(root)
    problems = [f"changed: {f}" for f in recorded if f in current and current[f] != recorded[f]]
    problems += [f"missing: {f}" for f in recorded if f not in current]
    problems += [f"not in freeze: {f}" for f in current if f not in recorded]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check the code freeze record.")
    parser.add_argument("action", choices=["write", "check"])
    parser.add_argument("--image", default="", help="Docker image used for the official runs (write only)")
    args = parser.parse_args()
    freeze = REPO_ROOT / FREEZE_FILE
    if args.action == "write":
        write(REPO_ROOT, freeze, args.image)
        print(f"Wrote {FREEZE_FILE.as_posix()}")
        return 0
    problems = check(REPO_ROOT, freeze)
    print("\n".join(problems) if problems else f"All files match {FREEZE_FILE.as_posix()}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
