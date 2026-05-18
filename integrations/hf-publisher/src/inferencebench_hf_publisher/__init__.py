"""Hugging Face Hub publisher for InferenceBench envelopes.

Public API:

    from inferencebench_hf_publisher import (
        publish_envelope_to_hf,
        HfPublishResult,
        HfPublishError,
        HfRepoCollisionError,
        HfRateLimitError,
        compute_repo_id,
        slugify,
        render_envelope_readme,
    )

The publisher uploads a signed Envelope as a Hugging Face *dataset* repo
under ``Yobitel`` (the production org). See
``skills/hf-publishing/SKILL.md`` for the full design.
"""

from inferencebench_hf_publisher.publish import (
    HfPublishError,
    HfPublishResult,
    HfRateLimitError,
    HfRepoCollisionError,
    compute_repo_id,
    publish_envelope_to_hf,
    slugify,
)
from inferencebench_hf_publisher.readme import render_envelope_readme

__all__ = [
    "HfPublishError",
    "HfPublishResult",
    "HfRateLimitError",
    "HfRepoCollisionError",
    "compute_repo_id",
    "publish_envelope_to_hf",
    "render_envelope_readme",
    "slugify",
]
