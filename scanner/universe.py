"""Stock universe: Nifty 500 (primary) and the full Excel list (optional).

`build_universe()` is a one-time/refresh step that writes the CSVs under data/.
`load_universe(scope)` is used by the app at runtime.
"""
from __future__ import annotations

import csv
import io
import os
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
NIFTY500_CSV = os.path.join(DATA_DIR, "nifty500.csv")
SYMBOLS_CSV = os.path.join(DATA_DIR, "symbols.csv")

NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
EXCEL_NAME = "NSE_INTRADAY_SCANNER_ENHANCED.xlsx"


def to_yahoo(symbol: str) -> str:
    """NSE symbol -> Yahoo Finance ticker."""
    return f"{symbol.strip().upper()}.NS"


def _fetch_nifty500_rows() -> list[dict]:
    req = urllib.request.Request(
        NIFTY500_URL,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"},
    )
    text = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        sym = (r.get("Symbol") or "").strip()
        if not sym:
            continue
        rows.append(
            {
                "SYMBOL": sym,
                "NAME": (r.get("Company Name") or "").strip(),
                "INDUSTRY": (r.get("Industry") or "").strip(),
            }
        )
    return rows


def _read_excel_symbols() -> list[dict]:
    import openpyxl

    path = os.path.join(HERE, "..", EXCEL_NAME)
    if not os.path.exists(path):
        path = os.path.join(HERE, EXCEL_NAME)
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb["📊 FULL DATA"]
    rows = []
    first = True
    for row in ws.iter_rows(values_only=True):
        if first:
            first = False
            continue
        sym = row[0]
        name = row[2] if len(row) > 2 else ""
        if sym:
            rows.append({"SYMBOL": str(sym).strip(), "NAME": str(name or "").strip(),
                         "INDUSTRY": ""})
    wb.close()
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["SYMBOL", "NAME", "INDUSTRY"])
        w.writeheader()
        w.writerows(rows)


def build_universe(verbose: bool = True) -> None:
    """Fetch the Nifty 500 list and extract the full Excel list; write both CSVs.

    Falls back gracefully: if the live Nifty 500 list can't be fetched we reuse a
    previously saved nifty500.csv (if present).
    """
    # Full Excel universe (optional "all" scope)
    try:
        excel_rows = _read_excel_symbols()
        _write_csv(SYMBOLS_CSV, excel_rows)
        if verbose:
            print(f"symbols.csv: {len(excel_rows)} rows")
    except Exception as e:  # pragma: no cover
        excel_rows = []
        if verbose:
            print(f"WARN could not read Excel symbols: {e}")

    # Nifty 500 (primary)
    try:
        rows = _fetch_nifty500_rows()
        _write_csv(NIFTY500_CSV, rows)
        if verbose:
            print(f"nifty500.csv: {len(rows)} rows (fetched live)")
    except Exception as e:
        if os.path.exists(NIFTY500_CSV):
            if verbose:
                print(f"WARN live Nifty500 fetch failed ({e}); keeping existing CSV")
        else:
            raise RuntimeError(
                "Could not fetch Nifty 500 list and no cached copy exists."
            ) from e


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_universe(scope: str = "nifty500") -> list[dict]:
    """Return [{'symbol','name','industry','ticker'}, ...] for the given scope."""
    path = NIFTY500_CSV if scope == "nifty500" else SYMBOLS_CSV
    if not os.path.exists(path):
        build_universe(verbose=False)
    out = []
    for r in _load_csv(path):
        sym = r["SYMBOL"].strip()
        out.append(
            {
                "symbol": sym,
                "name": r.get("NAME", "").strip(),
                "industry": r.get("INDUSTRY", "").strip(),
                "ticker": to_yahoo(sym),
            }
        )
    return out


if __name__ == "__main__":
    build_universe()
