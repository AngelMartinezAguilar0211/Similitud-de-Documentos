import argparse
import logging
from .http import HttpClient
from .arxiv import collect_arxiv, save_arxiv_corpus
from .pubmed import collect_pubmed_html, save_pubmed_corpus

def main():
    ap = argparse.ArgumentParser(description="Practice II - Web Scraping (arXiv & PubMed)")
    ap.add_argument("--repo", choices=["arxiv", "pubmed", "both"], default="both")
    ap.add_argument("--debug", action="store_true", help="Habilita logging DEBUG")

    ap.add_argument("--arxiv-per-section", type=int, default=100,
                    help="Número exacto por sección (cs.CL, cs.CV, cs.CR)")

    ap.add_argument("--pubmed-total", type=int, default=300,
                    help="Número exacto total (se paginará hasta alcanzarlo)")
    ap.add_argument("--pubmed-page-size", type=int, default=100,
                    help="Tamaño por página (recomendado 100)")

    ap.add_argument("--arxiv-out", default="arxiv_raw_corpus.csv")
    ap.add_argument("--pubmed-out", default="pubmed_raw_corpus.csv")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    http = HttpClient()

    if args.repo in ("arxiv", "both"):
        print(f"[arXiv] Recolectando {args.arxiv_per_section} por sección...")
        arxiv_rows = collect_arxiv(http, per_section_exact=args.arXiv_per_section if hasattr(args, 'arXiv_per_section') else args.arxiv_per_section)
        save_arxiv_corpus(arxiv_rows, args.arxiv_out)
        print(f"[arXiv] Guardado en {args.arxiv_out} ({len(arxiv_rows)} filas)")

    if args.repo in ("pubmed", "both"):
        print(f"[PubMed] Recolectando {args.pubmed_total} (páginas de {args.pubmed_page_size})...")
        pubmed_rows = collect_pubmed_html(http, required_total=args.pubmed_total, page_size=args.pubmed_page_size)
        save_pubmed_corpus(pubmed_rows, args.pubmed_out)
        print(f"[PubMed] Guardado en {args.pubmed_out} ({len(pubmed_rows)} filas)")

if __name__ == "__main__":
    main()
