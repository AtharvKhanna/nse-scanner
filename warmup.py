"""Pre-load the deployed Streamlit app's data so users don't wait on first open.

Run by a GitHub Actions cron at ~3:50 PM IST. It opens the public app in a real
headless browser (which makes Streamlit actually run the script and fetch + cache
the day's data), waking the app first if it's asleep.
"""
import os
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://nsescanner123.streamlit.app/")
MAX_WAIT = int(os.environ.get("MAX_WAIT", "330"))  # seconds to wait for data

# Any of these appearing means the scan finished and data is cached.
LOADED_MARKERS = ["auto-updates after market close", "Stocks shown", "Buy now",
                  "Last updated"]
WAKE_MARKERS = ["get this app back up", "Yes, get this app back up", "is asleep"]


def _visible_text(page):
    try:
        return page.locator("body").inner_text(timeout=5000)
    except Exception:
        return ""


def _try_wake(page):
    """If the Streamlit sleep screen is shown, click the wake button."""
    txt = _visible_text(page).lower()
    if any(m.lower() in txt for m in WAKE_MARKERS):
        print("App is asleep — attempting to wake it")
        for sel in ['button:has-text("back up")', 'text=/back up/i',
                    'button:has-text("Yes")']:
            try:
                el = page.locator(sel).first
                if el.count() > 0:
                    el.click(timeout=5000)
                    print(f"Clicked wake control: {sel}")
                    page.wait_for_timeout(10000)
                    return True
            except Exception as e:
                print("wake click failed:", sel, e)
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Opening {URL}")
        page.goto(URL, timeout=120000, wait_until="domcontentloaded")
        page.wait_for_timeout(8000)

        # Wake the app if needed (retry a couple of times).
        for _ in range(3):
            if _try_wake(page):
                page.wait_for_timeout(8000)
            else:
                break

        # Poll until a "data loaded" marker shows up.
        deadline = time.time() + MAX_WAIT
        loaded = False
        while time.time() < deadline:
            _try_wake(page)  # in case it fell back to sleep screen
            txt = _visible_text(page)
            if any(m.lower() in txt.lower() for m in LOADED_MARKERS):
                loaded = True
                break
            time.sleep(5)

        page.wait_for_timeout(6000)  # let the server finish writing its cache
        print(f"Data loaded: {loaded}")
        if not loaded:
            print("PAGE TITLE:", page.title())
            print("VISIBLE TEXT (first 800 chars):")
            print(_visible_text(page)[:800])
        browser.close()


if __name__ == "__main__":
    main()
