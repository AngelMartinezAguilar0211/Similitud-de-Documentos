import csv
from pathlib import Path
from typing import Iterable, List
from datetime import datetime

def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_tsv(path: str | Path, rows: Iterable[List[str]], header: List[str]) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for r in rows:
            cleaned = []
            for x in r:
                if x is None:
                    cleaned.append("")
                else:
                    cleaned.append(str(x).replace("\n", " ").replace("\r", " ").strip())
            row = cleaned + [""] * (len(header) - len(cleaned))
            writer.writerow(row)

def ddmmyyyy(date_like: str) -> str:
    if not date_like:
        return ""

    s = date_like.strip().replace(",", " ").replace("T", " ").replace("Z", "")
    candidates = [
        ("%Y-%m-%d", "%d/%m/%Y"),
        ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y"),
        ("%Y-%m-%d %H:%M", "%d/%m/%Y"),
        ("%Y-%m-%d %H", "%d/%m/%Y"),
        ("%d %b %Y", "%d/%m/%Y"),
        ("%b %d %Y", "%d/%m/%Y"),
        ("%d %B %Y", "%d/%m/%Y"),
        ("%B %d %Y", "%d/%m/%Y"),
        ("%Y %b %d", "%d/%m/%Y"),
        ("%Y %B %d", "%d/%m/%Y"),
        ("%Y", "%d/%m/%Y"),
    ]
    for inp, out in candidates:
        try:
            dt = datetime.strptime(s, inp)
            if inp == "%Y":
                dt = dt.replace(month=1, day=1)
            return dt.strftime(out)
        except Exception:
            continue
    try:
        dt = datetime.strptime(s, "%d/%m/%Y")
        return dt.strftime("%d/%m/%Y")
    except Exception:
        if s.isdigit() and len(s) == 4:
            return f"01/01/{s}"
        return ""

def normalize_authors(auth_list: List[str]) -> str:
    cleaned = [a.strip() for a in auth_list if a and a.strip()]
    return ", ".join(cleaned)
