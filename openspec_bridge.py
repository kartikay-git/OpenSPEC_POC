import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLANNING = ROOT / "planning"

CLI = (
    ROOT
    / "node_modules"
    / "@fission-ai"
    / "openspec"
    / "bin"
    / "openspec.js"
)

# Generation order: proposal -> specs -> design -> tasks.
ARTIFACTS = {
    "proposal": "proposal.md",
    "specs": "specs/greeting/spec.md",
    "design": "design.md",
    "tasks": "tasks.md",
}


def run_cli(*args: str, check: bool = True):
    node = shutil.which("node")

    if not node or not CLI.is_file():
        raise RuntimeError(
            "Install Node.js and run npm.cmd install."
        )

    if not (PLANNING / "openspec/config.yaml").is_file():
        raise RuntimeError(
            "Run npm.cmd run init:spec first."
        )

    environment = {
        **os.environ,
        "NO_COLOR": "1",
        "OPENSPEC_TELEMETRY": "0",
    }

    result = subprocess.run(
        [node, str(CLI), *args],
        cwd=PLANNING,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        shell=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"OpenSpec failed:\n{result.stdout}\n{result.stderr}"
        )

    return result


def cli_json(*args: str) -> dict:
    result = run_cli(*args, "--json")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Expected OpenSpec JSON, received:\n{result.stdout}"
        ) from exc


def new_change(name: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise ValueError(
            "Use a lowercase kebab-case change name."
        )

    # This example starts with no existing baseline specifications.
    specs_directory = PLANNING / "openspec/specs"

    if any(specs_directory.rglob("*.md")):
        raise RuntimeError(
            "This POC expects an empty baseline specs directory."
        )

    target = PLANNING / "openspec/changes" / name

    if not target.resolve().is_relative_to(PLANNING.resolve()):
        raise RuntimeError(
            "Change directory escapes the planning workspace."
        )

    if target.exists() or target.is_symlink():
        raise RuntimeError(
            "Change already exists. Use --change with a new name."
        )

    run_cli(
        "new",
        "change",
        name,
        "--schema",
        "spec-driven",
    )

    return target.resolve()


def get_status(name: str) -> dict:
    return cli_json("status", "--change", name)


def get_instructions(name: str, artifact: str) -> dict:
    if artifact not in ARTIFACTS:
        raise ValueError("Unsupported artifact.")

    return cli_json(
        "instructions",
        artifact,
        "--change",
        name,
    )


def validate_change(name: str) -> dict:
    result = run_cli(
        "validate",
        name,
        "--strict",
        "--no-interactive",
        "--json",
        check=False,
    )

    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }