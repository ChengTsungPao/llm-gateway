"""Run the gateway: `uv run llm-gateway [--config config.json]`."""

import argparse
import sys
from pathlib import Path

import uvicorn

from .config import ConfigError, load_config
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-gateway")
    parser.add_argument("--config", default="config.json", help="path to config.json (default: ./config.json)")
    args = parser.parse_args()
    try:
        cfg = load_config(Path(args.config).resolve())
    except ConfigError as e:
        sys.exit(f"config error: {e}")
    print(f"llm-gateway: {len(cfg.profiles)} models, UI and API on http://{cfg.host}:{cfg.port}")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
