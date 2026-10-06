"""Offline source/record contracts. No task execution or production release."""
from .adapters import adapt_rebench, adapt_smith
from .records import (
    ContractError, Record, TestPlan, Snapshots, Environment, Provenance,
    prepare_request, record_fact, load_fact, validate_fact, actor_view,
    publication_guard,
)
from .derive import derive_reward, derive_sft_eligibility, qualify

__all__ = [
    'ContractError', 'Record', 'TestPlan', 'Snapshots', 'Environment',
    'Provenance', 'adapt_rebench', 'adapt_smith', 'prepare_request',
    'record_fact', 'load_fact', 'validate_fact', 'actor_view',
    'publication_guard', 'derive_reward', 'derive_sft_eligibility', 'qualify',
]
