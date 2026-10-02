"""Per-account isolation for the Colab CLI.

``colab_cli`` hardcodes its OAuth token at ``~/.config/colab-cli/token.json`` and
offers no flag or environment override, so the only way to drive several Google
accounts from one machine is to give each account its own ``HOME``. Every helper
here builds that environment; nothing else in the controller touches ``HOME``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_ACCOUNT = "default"

# Colab's own public installed-app client, shipped in colab_cli/oauth_config.json.
# The "secret" is public by design for installed apps; the paired landing page is
# the only redirect URI Google accepts for it.
OAUTH_CLIENT_ID = (
    "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
)
OAUTH_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"
SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
)
ACCOUNT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ACCOUNTS_DIR_VARIABLE = "METRICDP_COLAB_ACCOUNTS_DIR"


def accounts_root() -> Path:
    """Directory holding one fake ``HOME`` per non-default account."""
    configured = os.environ.get(_ACCOUNTS_DIR_VARIABLE)
    root = Path(configured) if configured else Path.home() / ".colab-accounts"
    return root.expanduser()


def validate_account(account: str) -> str:
    if not ACCOUNT_PATTERN.fullmatch(account):
        raise ValueError(
            f"Invalid account name {account!r}: use letters, digits, '_', '.' or '-'"
        )
    return account


def account_home(account: str) -> Path:
    """The ``HOME`` that isolates one account's Colab token and session state."""
    validate_account(account)
    if account == DEFAULT_ACCOUNT:
        return Path.home()
    return accounts_root() / account


def config_dir(account: str) -> Path:
    return account_home(account) / ".config" / "colab-cli"


def token_path(account: str) -> Path:
    return config_dir(account) / "token.json"


def email_cache_path(account: str) -> Path:
    """Where we cache the Google address behind a token.

    ``token.json`` often stores an empty ``account`` field, so the address is
    resolved once at login and cached beside the token.
    """
    return config_dir(account) / "metricdp-account.json"


def account_env(account: str) -> dict[str, str]:
    """Environment for a ``colab`` invocation belonging to ``account``."""
    environment = dict(os.environ)
    environment["HOME"] = str(account_home(account))
    return environment


def prepare_account(account: str) -> Path:
    """Create the account's config directory so the CLI can write its token."""
    directory = config_dir(account)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_logged_in(account: str) -> bool:
    """Whether this account holds a refreshable Colab token."""
    return bool(_read_json(token_path(account)).get("refresh_token"))


def account_email(account: str) -> str | None:
    """Google address behind the stored token, if it is known."""
    cached = _read_json(email_cache_path(account)).get("email")
    if cached:
        return str(cached)
    email = _read_json(token_path(account)).get("account")
    return str(email) if email else None


def _access_token(account: str) -> str | None:
    """A usable access token, refreshing the stored one when it has expired.

    The refreshed token is deliberately not written back: ``colab_cli`` owns
    ``token.json`` and must stay the only writer.
    """
    token = _read_json(token_path(account))
    access = token.get("token")
    expiry = str(token.get("expiry") or "")
    if access and expiry:
        try:
            moment = datetime.fromisoformat(expiry)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            if moment > datetime.now(UTC):
                return str(access)
        except ValueError:
            return str(access)
    refresh = token.get("refresh_token")
    if not refresh:
        return str(access) if access else None
    payload = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": token.get("client_id", ""),
            "client_secret": token.get("client_secret", ""),
        }
    ).encode()
    request = urllib.request.Request(
        str(token.get("token_uri") or "https://oauth2.googleapis.com/token"),
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            refreshed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return str(access) if access else None
    return refreshed.get("access_token") or (str(access) if access else None)


def resolve_email(account: str) -> str | None:
    """Look the address up from Google once and cache it; failures are non-fatal."""
    known = account_email(account)
    if known:
        return known
    token = _access_token(account)
    if not token:
        return None
    request = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    email = payload.get("email")
    if not email:
        return None
    email_cache_path(account).write_text(
        json.dumps({"email": email}) + "\n", encoding="utf-8"
    )
    return str(email)


def pending_login_path(account: str) -> Path:
    return config_dir(account) / "metricdp-login.json"


def begin_login(account: str) -> str:
    """Start a copy-paste OAuth flow and return the URL to approve.

    The PKCE verifier is stored locally, so only this machine can redeem the
    code the approver reads back. Nothing secret travels in the URL, which makes
    it safe to hand to whoever owns the account being added.
    """
    prepare_account(account)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    state = secrets.token_urlsafe(16)
    pending_login_path(account).write_text(
        json.dumps({"verifier": verifier, "state": state, "started_at": _now()}) + "\n",
        encoding="utf-8",
    )
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": OAUTH_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": " ".join(SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "consent",
            "token_usage": "remote",
            "access_type": "offline",
        }
    )
    return f"{AUTH_URI}?{query}"


def finish_login(account: str, code: str) -> str | None:
    """Redeem an authorization code and write colab_cli's token file."""
    pending = _read_json(pending_login_path(account))
    verifier = pending.get("verifier")
    if not verifier:
        raise RuntimeError(
            f"No pending login for account {account!r}; start one with --begin."
        )
    payload = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code.strip(),
            "client_id": OAUTH_CLIENT_ID,
            "client_secret": OAUTH_CLIENT_SECRET,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        }
    ).encode()
    request = urllib.request.Request(
        TOKEN_URI,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            granted = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"Authorization code was rejected: {detail}") from None
    if not granted.get("refresh_token"):
        raise RuntimeError(
            "Google returned no refresh token; the code was probably already used."
        )
    expiry = datetime.now(UTC) + timedelta(seconds=int(granted.get("expires_in", 3600)))
    token = {
        "token": granted["access_token"],
        "refresh_token": granted["refresh_token"],
        "token_uri": TOKEN_URI,
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "scopes": list(SCOPES),
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": expiry.replace(tzinfo=None).isoformat() + "Z",
    }
    path = token_path(account)
    path.write_text(json.dumps(token) + "\n", encoding="utf-8")
    path.chmod(0o600)
    pending_login_path(account).unlink(missing_ok=True)
    return resolve_email(account)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def known_accounts() -> list[str]:
    """Every account the controller can use, default first."""
    accounts = [DEFAULT_ACCOUNT]
    root = accounts_root()
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name == DEFAULT_ACCOUNT:
                continue
            if ACCOUNT_PATTERN.fullmatch(entry.name):
                accounts.append(entry.name)
    return accounts


def logged_in_accounts() -> list[str]:
    return [account for account in known_accounts() if is_logged_in(account)]
