from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train COGNATE on one automatically selected device."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    device_mode = str(config.get("runtime", {}).get("device", "auto"))
    train(
        config_path,
        args.output,
        device_mode=device_mode,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
