from typing import List, Dict
from bs4 import BeautifulSoup
import re

from .http import HttpClient
from .io_utils import write_tsv, ddmmyyyy

PUBMED_BASE = "https://pubmed.ncbi.nlm.nih.gov"
TRENDING = f"{PUBMED_BASE}/trending/"

FIELD_KEYS = {"TI", "AB", "AU", "JT", "DP", "LID", "AID"}

_CAPS_COLON_ANYWHERE = re.compile(
    r"\b(?:[A-Z][A-Z0-9\-]{1,}(?:\s+[A-Z][A-Z0-9\-]{1,}){0,5}):\s*"
)

def normalize_authors(auth_list: List[str]) -> str:
    cleaned = [a.strip() for a in auth_list if a and a.strip()]
    return ", ".join(cleaned)

def clean_pubmed_abstract(text: str) -> str:
    return _CAPS_COLON_ANYWHERE.sub("", text or "").strip()

def _split_records(block_text: str) -> List[str]:
    lines = block_text.splitlines()
    records: List[List[str]] = []
    current: List[str] = []
    for ln in lines:
        if ln.startswith("PMID- "):
            if current:
                records.append(current)
                current = []
        current.append(ln)
    if current:
        records.append(current)
    return ["\n".join(r) for r in records]

def _extract_year_from_text(text: str) -> str:
    m = re.search(r"(19|20)\d{2}", text)
    if m:
        return f"01/01/{m.group(0)}"
    return ""

def _parse_medline_record(text: str) -> Dict[str, str | List[str]]:
    data: Dict[str, List[str]] = {k: [] for k in FIELD_KEYS}
    current_key = None
    for ln in text.splitlines():
        m = re.match(r"^([A-Z]{2,4})\s*-\s(.*)$", ln)
        if m:
            key, val = m.group(1).strip(), m.group(2).rstrip()
            current_key = key if key in FIELD_KEYS else None
            if current_key:
                data[current_key].append(val)
        else:
            if current_key in ("TI", "AB"):
                data[current_key].append(ln.rstrip())

    title = " ".join(data["TI"]).strip()
    abstract = " ".join(data["AB"]).strip()
    abstract = clean_pubmed_abstract(abstract)
    authors = [a.strip() for a in data["AU"] if a.strip()]
    journal = " ".join(data["JT"]).strip()
    date_raw = " ".join(data["DP"]).strip()

    def pick_doi(values: List[str]) -> str:
        for v in values:
            m = re.search(r"(10\.\d{4,9}/\S+)", v)
            if m:
                return m.group(1).rstrip(").;]")
        return ""

    doi = pick_doi(data["LID"]) or pick_doi(data["AID"])

    date_final = ddmmyyyy(date_raw)
    if not date_final:
        date_final = _extract_year_from_text(doi)
    if not date_final:
        date_final = _extract_year_from_text(journal)
    if not date_final:
        date_final = _extract_year_from_text(title) or _extract_year_from_text(abstract)

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "date": date_final,
        "doi": doi,
    }

def collect_pubmed_html(http: HttpClient, required_total: int = 300, page_size: int = 100) -> List[List[str]]:
    all_rows: List[List[str]] = []
    page = 1
    while len(all_rows) < required_total:
        url = (
            f"{TRENDING}"
            f"?term=&ac=yes&schema=none&page={page}"
            f"&show_snippets=on&sort=relevance&sort_order=desc"
            f"&format=pubmed&size={page_size}"
        )
        r = http.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        pre_blocks = soup.select("pre.search-results-chunk")
        if not pre_blocks:
            break

        for pre in pre_blocks:
            for rec_text in _split_records(pre.get_text("\n")):
                meta = _parse_medline_record(rec_text)
                if not meta["abstract"]:
                    continue
                if not meta["date"]:
                    continue
                if not meta["doi"]:
                    continue
                row = [
                    meta["doi"],
                    meta["title"],
                    normalize_authors(meta["authors"]),
                    meta["abstract"],
                    meta["journal"],
                    meta["date"],
                ]
                all_rows.append(row)
                if len(all_rows) >= required_total:
                    break
            if len(all_rows) >= required_total:
                break

        page += 1

    return all_rows[:required_total]

def save_pubmed_corpus(rows: List[List[str]], out_path: str = "pubmed_raw_corpus.tsv") -> None:
    header = ["DOI", "Title", "Authors", "Abstract", "Journal", "Date"]
    write_tsv(out_path, rows, header)
