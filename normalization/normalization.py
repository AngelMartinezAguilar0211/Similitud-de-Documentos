import argparse
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# ----------------------------- #
# Descargas necesarias
# ----------------------------- #
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# ----------------------------- #
# Funciones de utilidad
# ----------------------------- #
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def normalize_text_nltk(text: str) -> str:
    """
    Normaliza texto en inglés:
    - Convierte a minúsculas.
    - Tokeniza.
    - Etiqueta POS y lematiza.
    - Devuelve texto con tokens separados por espacios.
    """
    text = str(text).lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(tok, get_wordnet_pos(tag)) for tok, tag in tagged]
    return " ".join(lemmatized)


def normalize_single_text(text: str) -> str:
    """
    Aplica la misma normalización que se usa en los corpus, pero sobre una sola cadena.
    """
    return normalize_text_nltk(text)


def normalize_corpus(input_file: str, output_file: str):
    df = pd.read_csv(input_file, sep="\t")

    cols_to_normalize = [col for col in ["Title", "Abstract"] if col in df.columns]
    if not cols_to_normalize:
        raise ValueError("No se encontraron columnas 'Title' ni 'Abstract' en el archivo.")

    for col in cols_to_normalize:
        print(f"🔄 Normalizando columna: {col} ...")
        df[col] = df[col].astype(str).apply(normalize_text_nltk)

    df.to_csv(output_file, sep="\t", index=False)
    print(f"✅ Archivo normalizado guardado en: {output_file}")


# ----------------------------- #
# Argparse principal
# ----------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Normalización de texto (NLTK) para columnas 'Title' y 'Abstract', manteniendo puntuación."
    )
    parser.add_argument("--input", required=True, help="Ruta del archivo de entrada (.csv o .tsv).")
    parser.add_argument("--output", required=True, help="Ruta del archivo de salida (.csv o .tsv).")
    args = parser.parse_args()

    normalize_corpus(args.input, args.output)


if __name__ == "__main__":
    main()
