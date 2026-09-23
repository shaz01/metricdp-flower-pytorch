from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from scripts.colab import token_refresher as refresher
from scripts.colab.token_refresher import Listed


def test_plan_refresh_updates_only_changed_matching_endpoints():
    stored = {
        "a": ("ep-a", "old", "https://a"),
        "b": ("ep-b", "same", "https://b"),
        "gone": ("ep-x", "old", "https://x"),
    }
    listed = [
        Listed("ep-a", "new", "https://a", 3600),
        Listed("ep-b", "same", "https://b", 1200),
        Listed("ep-c", "other", "https://c", 3600),
    ]
    assert refresher.plan_refresh(stored, listed) == {"a": listed[0]}


def test_plan_refresh_ignores_empty_live_token():
    stored = {"a": ("ep-a", "old", "https://a")}
    assert refresher.plan_refresh(stored, [Listed("ep-a", "", "https://a")]) == {}


def test_plan_reattach_requires_single_unambiguous_pair():
    unnamed = Listed("ep-new", "tok", "https://new")
    named = Listed("ep-a", "tok", "https://a")
    assert refresher.plan_reattach({"a"}, {"ep-a"}, {"a", "b"}, [named, unnamed]) == ("b", unnamed)
    assert refresher.plan_reattach({"a"}, {"ep-a"}, {"a"}, [named, unnamed]) is None
    assert refresher.plan_reattach(set(), set(), {"a", "b"}, [unnamed]) is None
    assert refresher.plan_reattach(set(), set(), {"b"}, [unnamed, Listed("ep-2", "t", "u")]) is None


def test_controller_active_sessions_filters_account_and_phase(tmp_path: Path):
    for name, account, phase in [
        ("s1", "lab2", "training"),
        ("s2", "lab2", "collected"),
        ("s3", "default", "training"),
        ("s4", "lab2", "collecting"),
    ]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"session": name, "account": account, "phase": phase})
        )
    (tmp_path / "bad.json").write_text("{")
    assert refresher.controller_active_sessions(tmp_path, "lab2") == {"s1", "s4"}


class _Session:
    def __init__(self, name, endpoint, token, url):
        self.name, self.endpoint, self.token, self.url = name, endpoint, token, url


class _Store:
    def __init__(self, sessions):
        self.sessions = sessions
        self.saved = 0

    def list(self):
        return dict(self.sessions)

    @contextmanager
    def _lock_exclusive(self):
        yield None

    def _load_raw(self, handle):
        return self.sessions

    def _save_raw(self, handle, sessions):
        self.saved += 1
        self.sessions = sessions


def test_cycle_rewrites_stale_token_in_place():
    info = types.SimpleNamespace(token="fresh", url="https://u2", token_expires_in_seconds=3599)
    client = types.SimpleNamespace(
        list_assignments=lambda: [types.SimpleNamespace(endpoint="ep", runtime_proxy_info=info)]
    )
    store = _Store({"run": _Session("run", "ep", "stale", "https://u1")})
    state = types.SimpleNamespace(client=client, store=store)
    refresher.cycle(state, None, "default", reattach=False)
    assert (store.sessions["run"].token, store.sessions["run"].url) == ("fresh", "https://u2")
    assert store.saved == 1
    refresher.cycle(state, None, "default", reattach=False)
    assert store.saved == 1


def test_main_survives_api_error(monkeypatch, capsys):
    def boom():
        raise RuntimeError("503")

    fake_state = types.SimpleNamespace(client=types.SimpleNamespace(list_assignments=boom))
    module = types.ModuleType("colab_cli.common")
    module.state = fake_state
    monkeypatch.setitem(sys.modules, "colab_cli", types.ModuleType("colab_cli"))
    monkeypatch.setitem(sys.modules, "colab_cli.common", module)
    refresher.main(["--once"])
    assert "error RuntimeError: 503" in capsys.readouterr().out
