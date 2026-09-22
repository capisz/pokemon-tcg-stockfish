"""Research-only hierarchical policy learning.

Nothing in this package is imported by the production policy path.  Promotion
is deliberately a separate, human-approved operation.
"""

from .schema import LIMITS, SCHEMA_VERSION, IdentityManifest, identity_hash

__all__ = ["LIMITS", "SCHEMA_VERSION", "IdentityManifest", "identity_hash"]
