"""
Resilient HTTP client with manual retry, rate limiting, and error handling.

Handles HTTP 429 (rate limit), 5xx (server error), connection errors, and
timeouts with exponential backoff. This replaces tenacity for finer-grained
control over retry-after headers and error classification.
"""

import logging
import time
from typing import Any

import requests

from pipeline.config import MAX_RETRIES, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class ResilientClient:
    """HTTP client with automatic retry, rate limiting, and error handling."""

    def __init__(
        self,
        base_url: str = "",
        delay: float = 0.26,
        default_params: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.default_params = default_params or {}
        self.default_headers = default_headers or {}

        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            **self.default_headers,
        })

        self.request_count = 0
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make a GET request with retry, backoff, and rate limiting.

        Retries on:
          - HTTP 429 (reads Retry-After header)
          - HTTP 5xx (exponential backoff)
          - Connection errors / timeouts (exponential backoff)

        Raises on:
          - Non-retryable HTTP errors (4xx except 429)
          - Exhausted retries
        """
        url = (
            endpoint
            if endpoint.startswith("http://") or endpoint.startswith("https://")
            else (f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint)
        )

        merged_params = {**self.default_params}
        if params:
            merged_params.update(params)

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit()

                response = self.session.get(
                    url,
                    params=merged_params,
                    timeout=REQUEST_TIMEOUT,
                )

                self._last_request_time = time.monotonic()
                self.request_count += 1

                # --- Rate limiting ---
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", 10)
                    )
                    logger.warning(
                        "Rate limited (429). Waiting %ds "
                        "(attempt %d/%d)",
                        retry_after, attempt, MAX_RETRIES,
                    )
                    time.sleep(retry_after)
                    continue

                # --- Server errors ---
                if response.status_code >= 500:
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(
                        "Server error %d. Waiting %ds "
                        "(attempt %d/%d)",
                        response.status_code,
                        wait_time,
                        attempt,
                        MAX_RETRIES,
                    )
                    time.sleep(wait_time)
                    continue

                # --- Client errors (non-retryable) ---
                response.raise_for_status()
                return response.json()

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                wait_time = min(2 ** attempt, 60)
                logger.warning(
                    "Connection error: %s. Waiting %ds "
                    "(attempt %d/%d)",
                    type(exc).__name__,
                    wait_time,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(wait_time)

            except requests.HTTPError:
                # Non-retryable HTTP errors (4xx except 429)
                raise

        raise RuntimeError(
            f"Failed after {MAX_RETRIES} attempts: {url} "
            f"(last error: {last_error})"
        )

    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()
