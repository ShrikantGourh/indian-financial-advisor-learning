#!/usr/bin/env python3
"""Validate conversational JSONL datasets before training."""

import argparse
import json
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}
PROJECT_DIR = Path(__file__).resolve().parents[1]


def validate_file(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: expected a non-empty messages array")

            roles = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_number}: each message must be an object")
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or role not in ALLOWED_ROLES:
                    raise ValueError(f"{path}:{line_number}: unsupported role {role!r}")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_number}: message content must be non-empty text")
                roles.append(role)

            if "user" not in roles or "assistant" not in roles:
                raise ValueError(f"{path}:{line_number}: each record needs user and assistant messages")
            if roles[-1] != "assistant":
                raise ValueError(f"{path}:{line_number}: the final message must be from the assistant")
            count += 1

    if count == 0:
        raise ValueError(f"{path}: dataset contains no records")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[
            PROJECT_DIR / "data/train.jsonl",
            PROJECT_DIR / "data/validation.jsonl",
        ],
        help="JSONL files to validate (default: project train and validation files)",
    )
    args = parser.parse_args()

    total = 0
    for path in args.files:
        count = validate_file(path)
        total += count
        print(f"{path}: {count} valid record(s)")
    print(f"Validated {total} record(s).")


if __name__ == "__main__":
    main()
