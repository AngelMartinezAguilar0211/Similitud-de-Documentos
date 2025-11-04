import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Dict
from urllib.parse import urlencode

from .http import HttpClient
from .io_utils import write_tsv, normalize_authors, ddmmyyyy

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_SECTIONS = ["cs.CL", "cs.CV", "cs.CR"]
SECTION_LONG = {
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CR": "Cryptography and Security",
}

def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())

def _parse_atom(xml_text: str) -> List[Dict[str, object]]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    out: List[Dict[str, object]] = []
    root = ET.fromstring(xml_text)

    for entry in root.findall("atom:entry", ns):
        id_txt = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)$", id_txt)
        arxiv_id = m.group(1) if m else id_txt.rsplit("/", 1)[-1]

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()

        authors: List[str] = []
        for a in entry.findall("atom:author", ns):
            nm = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if nm:
                authors.append(nm)

        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

        doi = ""
        doi_el = entry.find("arxiv:doi", ns)
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        out.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "summary": summary,
            "published": published,
            "doi": doi,
        })
    return out

def _api_query(http: HttpClient, cat: str, start: int, max_results: int) -> List[Dict[str, object]]:
    params = {
        "search_query": f"cat:{cat}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    r = http.get(url, raise_for_status=True)
    log.debug("[arXiv API] %s -> %s", url, r.status_code)
    return _parse_atom(r.text)

def collect_arxiv(http: HttpClient, per_section_exact: int = 100, page_size: int = 200) -> List[List[str]]:
    all_rows: List[List[str]] = []

    for sec in ARXIV_SECTIONS:
        long_name = SECTION_LONG.get(sec, sec)
        rows: List[List[str]] = []
        seen = set()
        start = 0

        while len(rows) < per_section_exact:
            entries = _api_query(http, sec, start=start, max_results=page_size)
            if not entries:
                break

            for e in entries:
                if len(rows) >= per_section_exact:
                    break

                arxiv_id = e["id"]
                if arxiv_id in seen:
                    continue
                seen.add(arxiv_id)

                abstract = (e.get("summary") or "").strip()
                if not abstract:
                    continue  # abstract obligatorio

                title = (e.get("title") or "").strip()
                authors = normalize_authors([str(x) for x in (e.get("authors") or [])])
                raw_date = (e.get("published") or "").strip()
                date = ddmmyyyy(raw_date)
                doi = (e.get("doi") or "").strip()
                if not doi:
                    base_id = _strip_version(arxiv_id)
                    doi = f"10.48550/arXiv.{base_id}"

                rows.append([doi, title, authors, abstract, long_name, date])

            start += page_size

        if len(rows) < per_section_exact:
            log.warning("[arXiv API] WARNING: %s solo recolectó %d/%d", sec, len(rows), per_section_exact)

        all_rows.extend(rows[:per_section_exact])

    return all_rows

def save_arxiv_corpus(rows: List[List[str]], out_path: str = "arxiv_raw_corpus.csv") -> None:
    header = ["DOI", "Title", "Authors", "Abstract", "Section", "Date"]
    write_tsv(out_path, rows, header)
