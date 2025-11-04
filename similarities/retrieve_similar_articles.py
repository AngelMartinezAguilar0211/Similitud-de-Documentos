import argparse
import re
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from normalization.normalization import normalize_single_text  # usa la misma normalización NLTK

# ---------------------- #
#  Lectura de consulta   #
# ---------------------- #
def read_bibtex(file_path):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    title_match = re.search(r'title\s*=\s*[{"](.+?)[}"]', content, re.IGNORECASE)
    abstract_match = re.search(r'abstract\s*=\s*[{"](.+?)[}"]', content, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    return title, abstract


def read_ris(file_path):
    title, abstract = "", ""
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("TI  -"):
                title = line.replace("TI  -", "").strip()
            elif line.startswith("AB  -"):
                abstract = line.replace("AB  -", "").strip()
    return title, abstract


# ---------------------- #
#  Carga de archivos PKL #
# ---------------------- #
def ngram_code(ngram_type: str) -> str:
    if ngram_type == "unigram":
        return "n1-1"
    elif ngram_type == "bigram":
        return "n2-2"
    elif ngram_type == "both":
        return "n1-2"
    else:
        raise ValueError("Tipo de n-grama no reconocido.")


def load_pkl(base_path, corpus_name, field, vector_type, ngram_type):
    field = field.lower()
    ntag = ngram_code(ngram_type)
    vectors_dir = os.path.join(base_path, "data\\vectors")
    fname = f"{corpus_name}_{field}_{vector_type}_{ntag}.pkl"
    path = os.path.join(vectors_dir, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "vectorizer" not in data or "X" not in data:
        raise ValueError(f"El archivo {path} no contiene las claves esperadas ('vectorizer', 'X').")

    return data["vectorizer"], data["X"]


# ---------------------- #
#  Procesamiento general #
# ---------------------- #
def retrieve_similar_articles(query_file, field, vector_type, ngram_type, base_path, output_prefix):
    if query_file.endswith(".bib"):
        title, abstract = read_bibtex(query_file)
    elif query_file.endswith(".ris"):
        title, abstract = read_ris(query_file)
    else:
        raise ValueError("Formato no soportado. Usa .bib o .ris")

    query_text = title if field.lower() == "title" else abstract
    if not query_text:
        print(" No se encontró texto en el campo seleccionado.")
        return

    query_text = normalize_single_text(query_text)
    results = []

    for corpus_name in ["arxiv", "pubmed"]:
        try:
            vectorizer, X_corpus = load_pkl(base_path, corpus_name, field, vector_type, ngram_type)
        except (FileNotFoundError, ValueError) as e:
            print(f" {e}")
            continue

        csv_name = f"./data/corpus/{corpus_name}_raw_corpus.csv"
        csv_path = os.path.join(base_path, csv_name)
        if not os.path.exists(csv_path):
            print(f" No se encontró el archivo {csv_path}")
            continue

        corpus_df = pd.read_csv(csv_path, sep="\t")

        X_query = vectorizer.transform([query_text])
        similarities = cosine_similarity(X_query, X_corpus).flatten()

        top_indices = similarities.argsort()[::-1][:10]
        for idx in top_indices:
            row = corpus_df.iloc[idx]
            results.append({
                "Corpus": corpus_name,
                "Title": row["Title"],
                "DOI": row["DOI"],
                "Date": row.get("Date", "N/A"),
                "Similarity": similarities[idx]
            })

    results = sorted(results, key=lambda x: x["Similarity"], reverse=True)[:10]

    # ---------------------- #
    #  1) Salida de texto
    # ---------------------- #
    txt_path = f"{output_prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Archivo de consulta: {os.path.basename(query_file)}\n")
        f.write(f"Campo: {field} | Vectorización: {vector_type.upper()} | N-gramas: {ngram_type}\n\n")
        f.write("10 artículos más similares (ArXiv + PubMed):\n\n")
        for i, r in enumerate(results, start=1):
            f.write(f"{i}. [{r['Corpus'].upper()}] {r['Title']} (Similitud: {r['Similarity']:.3f})\n")
            f.write(f"   DOI: {r['DOI']}\n")
            f.write(f"   Fecha: {r['Date']}\n\n")
    print(f" Archivo TXT generado: {txt_path}")

    # ---------------------- #
    #  2) Salida TSV (interfaz)
    # ---------------------- #
    tsv_path = f"{output_prefix}.tsv"
    tsv_data = []
    for r in results:
        tsv_data.append({
            "CorpusDocument": r["Title"],
            "VectorRepresentation": vector_type,
            "ExtractedFeatures": ngram_type,
            "ComparisonContent": field,
            "SimilarityValue": round(r["Similarity"], 3),
            "Corpus": r["Corpus"],
            "DOI": r["DOI"],
            "Date": r["Date"],
        })

    df_tsv = pd.DataFrame(tsv_data)
    df_tsv.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
    print(f" Archivo TSV generado: {tsv_path}")


# ---------------------- #
#       ARGPARSE         #
# ---------------------- #
def main():
    parser = argparse.ArgumentParser(description="Recupera artículos similares y genera TXT + TSV (para interfaz gráfica).")
    parser.add_argument("--file", required=True, help="Archivo de consulta (.bib o .ris).")
    parser.add_argument("--field", choices=["Title", "Abstract"], default="Abstract", help="Campo a comparar (Title o Abstract).")
    parser.add_argument("--vector", choices=["tfidf", "frequency", "binary"], default="tfidf", help="Tipo de vectorización.")
    parser.add_argument("--ngrams", choices=["unigram", "bigram", "both"], default="unigram", help="Tipo de n-gramas (n1-1 / n2-2).")
    parser.add_argument("--basepath", default=".", help="Ruta base donde están los CSV crudos y la carpeta vectors/.")
    parser.add_argument("--output", default="similar_articles", help="Prefijo de los archivos de salida (sin extensión).")
    args = parser.parse_args()

    retrieve_similar_articles(
        query_file=args.file,
        field=args.field,
        vector_type=args.vector,
        ngram_type=args.ngrams,
        base_path=args.basepath,
        output_prefix=args.output
    )


if __name__ == "__main__":
    main()
