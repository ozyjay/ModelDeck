"""Build identity exposed by long-running ModelDeck services.

The Fedora package sets ``MODELDECK_BUILD_ID`` in its user services.  Reading it
once at import time means a running service continues to report the build that
started it, even if an RPM update replaces files on disk.
"""

from __future__ import annotations

import os

BUILD_ID = os.environ.get("MODELDECK_BUILD_ID", "development")
