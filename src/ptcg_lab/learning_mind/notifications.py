from __future__ import annotations

from types import MappingProxyType

ALLOWED_KINDS = frozenset({"pause", "failure", "review-ready", "milestone-complete"})
_VERIFIED_PROMOTION_TOKEN = object()


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
