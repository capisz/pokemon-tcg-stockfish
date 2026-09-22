from __future__ import annotations

ALLOWED_KINDS = frozenset({"pause", "failure", "review-ready", "milestone-complete"})


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

    def queue(self, candidate: str) -> None:
        self.candidate_checkpoint = candidate

    def promote(self, *, human_approved: bool, evidence_passed: bool) -> str:
        if not human_approved or not evidence_passed or not self.candidate_checkpoint:
            raise PermissionError("promotion requires passing evidence and explicit human approval")
        prior = self.trusted_checkpoint
        self.trusted_checkpoint = self.candidate_checkpoint; self.candidate_checkpoint = None
        return prior

    def rollback(self, prior: str) -> None:
        self.trusted_checkpoint = prior
