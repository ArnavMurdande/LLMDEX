"""
utils.py — Hardened Selenium driver factory and browser utilities.

SAFETY DESIGN:
    - All Selenium operations have explicit timeouts (page load, implicit wait,
      script timeout). The pipeline WILL NOT hang indefinitely.
    - Retry logic with exponential backoff for transient network failures.
    - Partial page load detection via DOM readiness checks.
    - Driver lifecycle is managed via context manager for guaranteed cleanup.
    - All exceptions are classified for the caller.
"""

from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration constants — tune these, don't scatter magic numbers
# ──────────────────────────────────────────────────────────────
PAGE_LOAD_TIMEOUT_SEC = 45
IMPLICIT_WAIT_SEC = 10
SCRIPT_TIMEOUT_SEC = 30
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SEC = 2.0


class ScraperBrowserError(Exception):
    """Raised when the browser cannot complete the requested operation."""
    pass


class PartialPageError(ScraperBrowserError):
    """Raised when the page loaded but critical DOM content is missing."""
    pass


def _build_chrome_options(headless: bool = True) -> Options:
    """Construct Chrome options with stability flags for CI environments."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")  # Modern headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    # Stable user-agent to avoid bot detection
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    return options


@contextmanager
def managed_driver(headless: bool = True) -> Generator[webdriver.Chrome, None, None]:
    """
    Context manager that guarantees driver cleanup.

    Usage:
        with managed_driver() as driver:
            driver.get("https://example.com")
            ...

    The driver is ALWAYS quit, even on exception.
    """
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        options = _build_chrome_options(headless)
        driver = webdriver.Chrome(service=service, options=options)

        # ── Timeout guards ──
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SEC)
        driver.implicitly_wait(IMPLICIT_WAIT_SEC)
        driver.set_script_timeout(SCRIPT_TIMEOUT_SEC)

        yield driver
    except WebDriverException as e:
        logger.error(f"WebDriver initialisation failed: {e}")
        raise ScraperBrowserError(f"Could not start Chrome: {e}") from e
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass  # Best-effort cleanup; don't mask the original error


def navigate_with_retry(
    driver: webdriver.Chrome,
    url: str,
    retries: int = MAX_RETRIES,
    wait_after_load_sec: float = 3.0,
) -> None:
    """
    Navigate to a URL with retry + exponential backoff.

    Raises ScraperBrowserError if all retries are exhausted.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Navigating to {url} (attempt {attempt}/{retries})")
            driver.get(url)
            # Wait for basic DOM readiness
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT_SEC).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            # Extra settle time for JS-heavy SPAs
            time.sleep(wait_after_load_sec)
            return
        except TimeoutException as e:
            last_error = e
            logger.warning(f"Page load timed out on attempt {attempt}: {e}")
        except WebDriverException as e:
            last_error = e
            logger.warning(f"WebDriver error on attempt {attempt}: {e}")

        if attempt < retries:
            backoff = RETRY_BACKOFF_BASE_SEC ** attempt
            logger.info(f"Retrying in {backoff:.1f}s...")
            time.sleep(backoff)

    raise ScraperBrowserError(
        f"Failed to load {url} after {retries} attempts: {last_error}"
    )


def scroll_to_bottom(driver: webdriver.Chrome, max_scrolls: int = 20, pause: float = 1.5) -> None:
    """
    Scroll down to trigger lazy-loading. Capped to avoid infinite loops.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def wait_for_element(
    driver: webdriver.Chrome,
    by,
    value: str,
    timeout: int = 15,
) -> None:
    """Wait until an element is present in the DOM. Raises TimeoutException."""
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )


def safe_float(text: str) -> Optional[float]:
    """
    Parse a string to float. Returns None (NOT 0) on failure.

    SAFETY: This is the ONLY numeric parser scrapers should use.
    """
    if text is None:
        return None
    import re
    cleaned = re.sub(r"[^\d.\-]", "", str(text).strip())
    if not cleaned or cleaned in (".", "-", "-."):
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def safe_int(text: str) -> Optional[int]:
    """Parse a string to int. Returns None (NOT 0) on failure."""
    f = safe_float(text)
    if f is None:
        return None
    return int(f)
