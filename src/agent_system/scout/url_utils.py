"""URL canonicalization — shared by scout/agent.py and scout/anthropic_source.py.

Deliberately its own module rather than living in ``scout/agent.py``:
``anthropic_source.py`` is documented as having zero dependency on
``scout/agent.py`` in either direction (see its module docstring), and
this is a small, generic, source-agnostic utility neither module
should have to duplicate — both importing it from a neutral third
module preserves that independence.

Not a general-purpose/RFC-complete URL canonicalizer — just enough to
collapse the common "same article, cosmetically different URL" cases
this project actually sees from RSS/sitemap sources: tracking query
params, a trailing slash, a fragment, or mixed scheme/host case.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Common tracking params worth stripping. Prefix-matched for the
# ubiquitous utm_* family; exact-matched for a handful of other common
# ones. Not exhaustive by design — see module docstring.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAM_NAMES = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "source"}


def canonicalize_url(url: str) -> str:
    """Normalize *url* so trivially-different links to the same article
    compare equal.

    * scheme/host lower-cased
    * fragment removed
    * trailing slash on the path removed (root "/" kept as-is)
    * common tracking query params removed
    * remaining query params sorted for stable ordering

    Returns "" for an empty/whitespace-only input. Never raises — a URL
    ``urlsplit`` can't parse is returned stripped but otherwise as-is,
    since dedup on a merely-imperfect key is still better than crashing
    the caller.
    """

    stripped = (url or "").strip()
    if not stripped:
        return ""

    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept_params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PARAM_PREFIXES)
        and key.lower() not in _TRACKING_PARAM_NAMES
    ]
    kept_params.sort()
    query = urlencode(kept_params)

    return urlunsplit((scheme, netloc, path, query, ""))
