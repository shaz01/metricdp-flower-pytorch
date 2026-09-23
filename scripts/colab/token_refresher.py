"""Keep Colab runtime-proxy tokens fresh for long-running controller sessions.

``colab_cli`` stores each session's runtime-proxy token once, at assignment, and
never refreshes it. The token expires after about an hour; the next ``colab exec``
then gets 404/401, prunes the session from ``sessions.json`` and kills its
keep-alive while the VM keeps training. Every controller probe or collection
after that fails.

This daemon runs beside the controller, one process per account, under the Colab
CLI's own interpreter and with that account's ``HOME``::

    HOME=<account home> <colab-cli python> scripts/colab/token_refresher.py \
        --account <name> --interval 120

Each cycle lists the account's live assignments (which carry a current token and
URL), and rewrites only the ``token``/``url`` of stored sessions whose endpoint
matches, under the CLI's own exclusive state lock. It never assigns, unassigns
or deletes anything. With ``--reattach`` it re-registers a controller session
that the CLI already pruned, but only when the match is unambiguous: exactly one
active controller session of this account is missing from the CLI store and
exactly one live assignment is unnamed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ACTIVE_PHASES = {"launching", "training", "collecting"}


@dataclass(frozen=True)
class Listed:
    endpoint: str
    token: str
    url: str
    expires_in: int | None = None


def plan_refresh(
    stored: dict[str, tuple[str, str, str]], listed: list[Listed]
) -> dict[str, Listed]:
    """Sessions whose stored ``(endpoint, token, url)`` differs from the live one."""
    by_endpoint = {item.endpoint: item for item in listed}
    updates = {}
    for name, (endpoint, token, url) in stored.items():
        live = by_endpoint.get(endpoint)
        if live is not None and live.token and (live.token != token or live.url != url):
            updates[name] = live
    return updates


def plan_reattach(
    stored_names: set[str],
    stored_endpoints: set[str],
    controller_active: set[str],
    listed: list[Listed],
) -> tuple[str, Listed] | None:
    """The single unambiguous (pruned session, unnamed assignment) pair, if any."""
    missing = sorted(controller_active - stored_names)
    unnamed = [item for item in listed if item.endpoint not in stored_endpoints]
    if len(missing) == 1 and len(unnamed) == 1:
        return missing[0], unnamed[0]
    return None


def controller_active_sessions(controller_dir: Path, account: str) -> set[str]:
    """Active controller sessions assigned to ``account``."""
    active = set()
    for path in controller_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("account", "default") == account and data.get("phase") in ACTIVE_PHASES:
            active.add(data.get("session") or path.stem)
    return active


def _log(message: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}", flush=True)


def _listed(client) -> list[Listed]:
    out = []
    for assignment in client.list_assignments():
        info = getattr(assignment, "runtime_proxy_info", None)
        if info is None:
            continue
        out.append(
            Listed(
                assignment.endpoint,
                info.token,
                info.url,
                getattr(info, "token_expires_in_seconds", None),
            )
        )
    return out


def cycle(state, controller_dir: Path | None, account: str, reattach: bool) -> None:
    listed = _listed(state.client)
    store = state.store
    sessions = store.list()
    stored = {name: (s.endpoint, s.token, s.url) for name, s in sessions.items()}
    for name, live in plan_refresh(stored, listed).items():
        with store._lock_exclusive() as handle:
            current = store._load_raw(handle)
            session = current.get(name)
            if session is None or session.endpoint != live.endpoint:
                continue
            session.token, session.url = live.token, live.url
            store._save_raw(handle, current)
        _log(f"refreshed {name} expires_in={live.expires_in}")
    for name, (endpoint, _, _) in stored.items():
        live = next((item for item in listed if item.endpoint == endpoint), None)
        _log(f"ok {name} live={live is not None} expires_in={live.expires_in if live else None}")
    if not (reattach and controller_dir):
        return
    pair = plan_reattach(
        set(sessions),
        {s.endpoint for s in sessions.values()},
        controller_active_sessions(controller_dir, account),
        listed,
    )
    if pair is None:
        return
    from colab_cli.commands.session import spawn_keep_alive
    from colab_cli.state import SessionState

    name, live = pair
    session = SessionState(
        name=name, token=live.token, url=live.url, endpoint=live.endpoint,
        variant="GPU", accelerator="A100",
    )
    store.add(session)
    session.keep_alive_pid = spawn_keep_alive(
        live.endpoint, name, auth_provider=state.auth_provider, config_path=state.config_path
    )
    store.add(session)
    _log(f"reattached {name} keep_alive_pid={session.keep_alive_pid}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--account", default="default")
    parser.add_argument("--interval", type=float, default=120)
    parser.add_argument("--controller-dir", type=Path)
    parser.add_argument("--reattach", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    from colab_cli.common import state

    while True:
        try:
            cycle(state, args.controller_dir, args.account, args.reattach)
        except Exception as error:  # keep refreshing through transient API errors
            _log(f"error {type(error).__name__}: {error}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
