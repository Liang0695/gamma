"""Executor boundary ONLY. This package provides no concrete runner."""
from typing import Protocol
from .records import Record


class ControlledRunner(Protocol):
    def execute(self, task: Record, request: Record) -> Record:
        """Return a fact from real observations; caller enforces resource/ACL gates.

        Required integration: resolve authorized refs, immutable trees/test plan,
        isolated workspace, identity attestation, timeout/cleanup and raw logs.
        There is deliberately no implementation or fake success fallback here.
        """
        ...
