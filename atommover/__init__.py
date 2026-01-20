"""Top-level package exports for atommover.

The original module eagerly imported the heavy ``atommover.utils.imaging``
sub-package, which in turn pulls in optional visualization dependencies such as
Matplotlib.  On minimal or headless environments these imports can fail (for
example, when NumPy/Matplotlib binary wheels are unavailable), preventing users
from accessing the core simulation and algorithm modules.

To make the package usable without the optional imaging stack, we now attempt
to import ``imaging`` lazily and tolerate ImportError by exposing ``None``
instead.  Downstream code can check for ``imaging is None`` before invoking the
visualization helpers, while algorithmic modules remain importable.
"""

from __future__ import annotations

from typing import Optional

try:
	from .utils import imaging
except Exception:  # pragma: no cover - best effort optional dependency
	imaging = None

__all__ = [
	"imaging",
]
