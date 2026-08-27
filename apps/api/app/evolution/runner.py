from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.evolution import DEFAULT_PROMPT, run_lineage_cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue a bounded project generation lineage from this workspace.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    continue_parser = subparsers.add_parser("continue")
    continue_parser.add_argument("--lineage-root", required=True)
    continue_parser.add_argument("--prompt-file", required=True)
    continue_parser.add_argument("--remaining-generations", type=int, required=True)
    continue_parser.add_argument("--mode", choices=["conservative", "experimental", "tests-only"], default="conservative")
    continue_parser.add_argument("--stop-on-failure", action="store_true")

    args = parser.parse_args()
    if args.command != "continue":
        parser.error("Unsupported command")

    prompt_path = Path(args.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else DEFAULT_PROMPT
    result = run_lineage_cli(
        parent_repo=Path.cwd(),
        lineage_root=Path(args.lineage_root),
        prompt=prompt or DEFAULT_PROMPT,
        mode=args.mode,
        stop_on_failure=bool(args.stop_on_failure),
        remaining_generations=max(0, int(args.remaining_generations)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
