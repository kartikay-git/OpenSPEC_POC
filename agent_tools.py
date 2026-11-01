import hashlib
import json
import re
from pathlib import Path
from threading import Lock
from langchain.tools import tool

from openspec_bridge import ARTIFACTS, ROOT


SOURCE = ROOT / "source/hello.py"


def source_hash() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def artifact_path(change_dir: Path, artifact: str) -> Path:
    target = change_dir / ARTIFACTS[artifact]

    for item in [target, *target.parents]:
        is_junction = (
            hasattr(item, "is_junction")
            and item.is_junction()
        )

        if item.is_symlink() or is_junction:
            raise ValueError("Linked output paths are not allowed.")

        if item == change_dir:
            break

    if not target.resolve().is_relative_to(change_dir.resolve()):
        raise ValueError("Output path escapes the change directory.")

    return target


def make_tools(
    change_dir: Path | None,
    artifact: str | None,
):
    evidence = {
        "source_read": False,
        "plans_read": False,
        "written": False,
    }

    write_lock = Lock()

    @tool
    def read_source() -> str:
        """Read source/hello.py and its SHA-256 hash without executing it."""
        content = SOURCE.read_text(encoding="utf-8")

        result = json.dumps({
            "path": "source/hello.py",
            "sha256": source_hash(),
            "content": content,
        })

        evidence["source_read"] = True
        print("  TOOL: read_source", flush=True)

        return result

    # Used by planner.py --check.
    if change_dir is None or artifact is None:
        return [read_source], evidence

    @tool
    def read_plans() -> str:
        """Read existing proposal, specs, design and tasks for this change."""
        documents = {}

        for key, relative_path in ARTIFACTS.items():
            path = artifact_path(change_dir, key)

            if path.is_file():
                documents[relative_path] = path.read_text(
                    encoding="utf-8"
                )

        result = json.dumps(documents)

        evidence["plans_read"] = True
        print("  TOOL: read_plans", flush=True)

        return result

    @tool
    def write_artifact(content: str) -> str:
        """Save complete Markdown for this stage's assigned planning artifact."""
        with write_lock:
            if evidence["written"]:
                return (
                    "This stage already saved its artifact. "
                    "No additional write was performed."
                )

            if not evidence["source_read"]:
                return "ERROR: Call read_source before writing."

            if not evidence["plans_read"]:
                return "ERROR: Call read_plans before writing."

            content = content.strip()

            if not 80 <= len(content) <= 24000:
                return (
                    "ERROR: Write a complete document between "
                    "80 and 24000 characters."
                )

            if content.startswith("```"):
                return (
                    "ERROR: Write Markdown directly. Do not wrap "
                    "the whole document in a code fence."
                )

            if artifact == "tasks":
                completed_task = re.search(
                    r"- \[[xX]\]",
                    content,
                )

                if "- [ ]" not in content or completed_task:
                    return (
                        "ERROR: Use unchecked '- [ ]' tasks only."
                    )

            path = artifact_path(change_dir, artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                content + "\n",
                encoding="utf-8",
            )

            evidence["written"] = True

            print(
                f"  TOOL: write_artifact -> {ARTIFACTS[artifact]}",
                flush=True,
            )

            return (
                f"Saved {ARTIFACTS[artifact]}. "
                "The controller will end this stage."
            )

    return [read_source, read_plans, write_artifact], evidence