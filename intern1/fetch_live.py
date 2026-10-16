"""Entry point for the live Reliance data fetcher.

Run (from this folder):

    uv run fetch_live.py
    uv run fetch_live.py --config config/data_config.yaml

Reads credentials from the environment / .env, loads the YAML config, opens a
KiteTicker session for the configured instrument, and appends one CSV row per
tick to ``{data_dir}/{date}_Relience.csv`` until market close.

Data capture only — this process never places orders.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Supporting packages live next to this script; make them importable whether run
# as `uv run fetch_live.py` or `python fetch_live.py`.
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.config import ConfigError, load_config, load_credentials  # noqa: E402
from core.instruments import InstrumentResolutionError  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from core.stream import LiveFetcher, SessionError  # noqa: E402
from writer import CsvTickWriter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Reliance tick fetcher (Kite).")
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "config" / "data_config.yaml"),
        help="Path to data_config.yaml (default: config/data_config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional path to a .env file (default: nearest .env on the path)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config, base_dir=BASE_DIR)
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    today = datetime.now(config.session.tz).date()
    setup_logging(config.logging.log_dir, config.logging.level, today=today)
    log = logging.getLogger("fetch_live")

    try:
        env_file = Path(args.env_file) if args.env_file else None
        credentials = load_credentials(env_file=env_file)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    writer = CsvTickWriter(
        config.output.data_dir,
        config.output.filename_for(datetime.now(config.session.tz)),
        fsync=config.write.fsync,
        flush_every_n_ticks=config.write.flush_every_n_ticks,
    )

    fetcher = LiveFetcher(config, credentials, writer)
    try:
        fetcher.prepare()
    except (SessionError, InstrumentResolutionError) as exc:
        log.error("Startup aborted: %s", exc)
        writer.close()
        return 1

    try:
        fetcher.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user; shutting down.")
    except Exception:  # noqa: BLE001
        log.exception("Fetcher stopped on an unexpected error")
        return 1
    finally:
        writer.close()

    log.info("Session ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
