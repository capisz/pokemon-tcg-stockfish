from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage
import fcntl
import json
import math
import os
from pathlib import Path
import smtplib
import ssl
import time
from types import MappingProxyType

ALLOWED_KINDS = frozenset({"pause", "failure", "review-ready", "milestone-complete"})
_LOCAL_EVENT_FIELDS = frozenset({"reason", "milestone", "candidateHash", "reportHash"})
_VERIFIED_PROMOTION_TOKEN = object()


def _notification_record(event: dict, clock) -> dict | None:
    if not isinstance(event, dict) or event.get("kind") not in ALLOWED_KINDS:
        return None
    now = clock()
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("notification clock must return a finite timestamp")
    record = {"timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"),
              "kind": event["kind"]}
    for key in _LOCAL_EVENT_FIELDS:
        value = event.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("notification fields must be strings")
        record[key] = value[:512]
    return record


class LocalJsonlNotificationSink:
    """Append minimal, private local operator events; never stores arbitrary payloads."""

    def __init__(self, path: Path, *, clock=time.time):
        self.path = Path(path)
        self.clock = clock

    def __call__(self, event: dict) -> None:
        record = _notification_record(event, self.clock)
        if record is None:
            return
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(encoded) > 2048:
            raise ValueError("local notification record exceeds its size limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short write while recording local notification")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class SmtpEmailNotificationSink:
    """Opt-in email delivery over mandatory STARTTLS; credentials are never persisted."""

    def __init__(self, *, host: str, port: int, sender: str, recipient: str,
                 username: str, password: str, clock=time.time, timeout: float = 10.0,
                 smtp_factory=smtplib.SMTP, ssl_context_factory=ssl.create_default_context):
        values = (host, sender, recipient, username, password)
        if (any(not isinstance(value, str) or not value or "\r" in value or "\n" in value
                for value in values)
                or type(port) is not int or not 1 <= port <= 65535
                or isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("SMTP notification configuration is invalid")
        self.host, self.port = host, port
        self.sender, self.recipient = sender, recipient
        self.username, self.password = username, password
        self.clock, self.timeout = clock, timeout
        self.smtp_factory, self.ssl_context_factory = smtp_factory, ssl_context_factory

    @classmethod
    def from_environment(cls, *, environ=None, clock=time.time):
        env = os.environ if environ is None else environ
        required = ("PTCG_LEARNING_SMTP_HOST", "PTCG_LEARNING_SMTP_SENDER",
                    "PTCG_LEARNING_SMTP_RECIPIENT", "PTCG_LEARNING_SMTP_USERNAME",
                    "PTCG_LEARNING_SMTP_PASSWORD")
        missing = [key for key in required if not env.get(key)]
        if missing:
            raise ValueError("SMTP notifications are not configured: " + ", ".join(missing))
        raw_port = env.get("PTCG_LEARNING_SMTP_PORT", "587")
        try:
            port = int(raw_port)
        except (TypeError, ValueError) as error:
            raise ValueError("SMTP notification port must be an integer") from error
        return cls(host=env[required[0]], port=port, sender=env[required[1]],
                   recipient=env[required[2]], username=env[required[3]],
                   password=env[required[4]], clock=clock)

    def __call__(self, event: dict) -> None:
        record = _notification_record(event, self.clock)
        if record is None:
            return
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = self.recipient
        message["Subject"] = f"Pokémon Learning Mind: {record['kind']}"
        message.set_content(json.dumps(record, sort_keys=True, indent=2))
        with self.smtp_factory(self.host, self.port, timeout=self.timeout) as client:
            client.ehlo()
            client.starttls(context=self.ssl_context_factory())
            client.ehlo()
            client.login(self.username, self.password)
            client.send_message(message)


class VerifiedPromotionEvidence:
    """Unforgeable in-process receipt for a separately audited promotion package."""
    __slots__ = ("_values",)

    def __init__(self, values: dict, *, _verification_token: object):
        if _verification_token is not _VERIFIED_PROMOTION_TOKEN:
            raise TypeError("promotion evidence must come from the source-artifact verifier")
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, _name, _value):
        raise AttributeError("verified promotion evidence is immutable")


class NotificationRouter:
    def __init__(self, *, local=None, email=None):
        self.local = local; self.email = email

    def __call__(self, event: dict) -> None:
        if event.get("kind") not in ALLOWED_KINDS:
            return
        if self.local: self.local(event)
        if self.email: self.email(event)


class AtomicRollbackRegistry:
    def __init__(self, trusted_checkpoint: str):
        self.trusted_checkpoint = trusted_checkpoint
        self.candidate_checkpoint: str | None = None
        self.rollback_checkpoint: str | None = None

    def queue(self, candidate: str) -> None:
        self.candidate_checkpoint = candidate

    def promote(self, *, human_approved: bool,
                evidence: VerifiedPromotionEvidence | None = None) -> str:
        if (human_approved is not True or not isinstance(evidence, VerifiedPromotionEvidence)
                or evidence._values.get("candidateCheckpoint") != self.candidate_checkpoint
                or evidence._values.get("promotionCriteriaPassed") is not True
                or evidence._values.get("automaticPromotion") is not False
                or not self.candidate_checkpoint):
            raise PermissionError("promotion requires verifier-issued evidence and explicit human approval")
        prior = self.trusted_checkpoint
        self.rollback_checkpoint = prior
        self.trusted_checkpoint = self.candidate_checkpoint; self.candidate_checkpoint = None
        return prior

    def rollback(self, prior: str) -> None:
        if not isinstance(prior, str) or prior != self.rollback_checkpoint:
            raise PermissionError("rollback requires the exact checkpoint retained by an accepted promotion")
        self.trusted_checkpoint = prior
        self.rollback_checkpoint = None
