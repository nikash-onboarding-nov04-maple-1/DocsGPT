"""Connect the existing OTel replay to Convergent without a second span pipeline."""

from __future__ import annotations

import os
from typing import Any

from docsgpt.version import get_version


def configured_provider() -> Any:
    """Initialize Convergent only when a destination is configured.

    Returns:
        The SDK's provider, or None to retain the application's OTel setup.
    """
    if not any(os.getenv(key) for key in (
        "CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER"
    )):
        return None
    import convergent

    status = convergent.init(release=os.getenv("CONVERGENT_RELEASE") or get_version())
    return convergent.tracer_provider() if status.enabled else None
