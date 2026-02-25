"""External sensor data ingestion clients."""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 20  # seconds — generous for slow Pi connections


def fetch_with_retry(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    retries: int = 2,
    backoff: float = 2.0,
) -> requests.Response:
    """GET *url* with automatic retry on transient failures.

    Retries on connection errors, timeouts, and 5xx responses.
    Raises on non-retryable errors (4xx) immediately.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):  # 1 initial + retries
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            log.warning("HTTP %d from %s (attempt %d/%d)",
                        resp.status_code, url[:80], attempt, retries + 1)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            log.warning("Network error on %s (attempt %d/%d): %s",
                        url[:80], attempt, retries + 1, exc)
        except requests.HTTPError:
            raise

        if attempt <= retries:
            wait = backoff * attempt
            time.sleep(wait)

    raise last_exc or requests.ConnectionError(f"Failed after {retries + 1} attempts")
