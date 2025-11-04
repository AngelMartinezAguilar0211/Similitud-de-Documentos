import argparse
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from normalization.normalization import normalize_text_nltk  

# ----------------------------- #
# Vectorizador base
# ----------------------------- #
def build_vectorizer(rep: str, ngram_range: tuple):
    """Crea un vectorizador según el tipo de representación."""
    token_pattern = r"(?u)\w+|\?|\.|,|\¿|\!"
    if rep == "tfidf":
        return TfidfVectorizer(token_pattern=token_pattern, ngram_range=ngram_range)
    elif rep == "frequency":
        return CountVectorizer(token_pattern=token_pattern, ngram_range=ngram_range)
    elif rep == "binary":
        return CountVectorizer(binary=True, token_pattern=token_pattern, ngram_range=ngram_range)
    else:
        raise ValueError(f"Tipo de vectorización no reconocido: {rep}")

# ----------------------------- #
# Vectorización por columna
# ----------------------------- #
def vectorize_column(df: pd.DataFrame, corpus_name: str, column: str, rep: str, ngram_range: tuple, outdir: str):
    ntag = f"n{ngram_range[0]}-{ngram_range[1]}"
    vec = build_vectorizer(rep, ngram_range)
    X = vec.fit_transform(df[column].fillna(""))
    features = vec.get_feature_names_out()

    os.makedirs(outdir, exist_ok=True)
    fname = f"{corpus_name}_{column.lower()}_{rep}_{ntag}.pkl"
    fpath = os.path.join(outdir, fname)

    payload = {
        "vectorizer": vec,
        "X": X,
        "feature_names": features,
        "doc_ids": df.index.copy(),
        "meta": {
            "corpus": corpus_name,
            "column": column,
            "rep": rep,
            "ngram_min": ngram_range[0],
            "ngram_max": ngram_range[1],
        },
    }

    with open(fpath, "wb") as f:
        pickle.dump(payload, f)

    print(f"✅ {corpus_name} | {column} | {rep.upper()} | {ntag} → {fpath}")
    print(f"   Documentos: {X.shape[0]} | Características: {X.shape[1]}")

# ----------------------------- #
# Vectorización general
# ----------------------------- #
def vectorize_corpus(basepath: str, corpus: str, field: str, rep: str, ngrams: str):
    ngram_ranges = {
        "unigram": [(1, 1)],
        "bigram": [(2, 2)],
        "both": [(1, 1), (2, 2)],
    }

    reps = [rep] if rep != "all" else ["tfidf", "frequency", "binary"]
    fields = [field] if field != "Both" else ["Title", "Abstract"]
    corpora = [corpus] if corpus != "both" else ["arxiv", "pubmed"]

    for corpus_name in corpora:
        csv_path = os.path.join(basepath, f"corpus\\{corpus_name}_raw_corpus.csv")
        if not os.path.exists(csv_path):
            print(f" No se encontró el archivo: {csv_path}")
            continue

        df = pd.read_csv(csv_path, sep="\t")
        df = df.fillna("")

        # 🔄 Normalizar texto según la configuración oficial (manteniendo puntuación)
        for col in fields:
            if col not in df.columns:
                print(f" Columna '{col}' no encontrada en {corpus_name}. Se omite.")
                continue

            print(f"\n🔹 Normalizando {corpus_name} [{col}]...")
            df[col] = df[col].astype(str).apply(normalize_text_nltk)

            for rep_type in reps:
                for ngmin, ngmax in ngram_ranges[ngrams]:
                    outdir = os.path.join(basepath, "vectors")
                    vectorize_column(df, corpus_name, col, rep_type, (ngmin, ngmax), outdir)

# ----------------------------- #
# Argparse principal
# ----------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Vectoriza corpus (ArXiv, PubMed o ambos) con distintas representaciones y campos."
    )
    parser.add_argument("--basepath", default=".", help="Ruta base donde están los corpus crudos y la carpeta vectors/.")
    parser.add_argument("--corpus", choices=["arxiv", "pubmed", "both"], default="both", help="Corpus a vectorizar.")
    parser.add_argument("--field", choices=["Title", "Abstract", "Both"], default="Both", help="Campo de texto a vectorizar.")
    parser.add_argument("--rep", choices=["tfidf", "frequency", "binary", "all"], default="all", help="Tipo de vectorización.")
    parser.add_argument("--ngrams", choices=["unigram", "bigram", "both"], default="both", help="Tipo de n-gramas.")
    args = parser.parse_args()

    vectorize_corpus(args.basepath, args.corpus, args.field, args.rep, args.ngrams)

if __name__ == "__main__":
    main()
