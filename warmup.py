"""Pre-load the deployed Streamlit app's data so users don't wait on first open.

Run by a GitHub Actions cron at ~3:50 PM IST. It opens the public app in a real
headless browser (which makes Streamlit actually run the script and fetch + cache
the day's data), waking the app first if it's asleep.
"""
import os
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://nsescanner123.streamlit.app/")
MAX_WAIT = int(os.environ.get("MAX_WAIT", "210"))  # seconds to wait for data


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Opening {URL}")
        page.goto(URL, timeout=120000, wait_until="domcontentloaded")

        # If the app is asleep, Streamlit shows a wake-up button — click it.
        for _ in range(3):
            try:
                btn = page.get_by_text("get this app back up", exact=False)
                if btn.count() > 0:
                    print("App was asleep — clicking wake button")
                    btn.first.click()
                    page.wait_for_timeout(8000)
                else:
                    break
            except Exception as e:
                print("wake check:", e)
                break

        # Wait until the scan finishes — the 'Last updated' caption appears once data loads.
        deadline = time.time() + MAX_WAIT
        loaded = False
        while time.time() < deadline:
            try:
                if page.get_by_text("Last updated", exact=False).count() > 0:
                    loaded = True
                    break
            except Exception:
                pass
            page.wait_for_timeout(3000)

        # Let the server finish writing its cache.
        page.wait_for_timeout(6000)
        print(f"Data loaded: {loaded}")
        browser.close()
        if not loaded:
            # don't hard-fail the job; the app may still have warmed
            print("WARNING: 'Last updated' not detected within timeout.")


if __name__ == "__main__":
    main()
