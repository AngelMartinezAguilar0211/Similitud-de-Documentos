# Sistema de Similitud de Documentos

Este proyecto implementa una **tubería completa de procesamiento de lenguaje natural (PLN)** para la **búsqueda, normalización, representación y comparación de artículos científicos** obtenidos de los repositorios **arXiv** y **PubMed**.  

Su propósito es identificar los documentos más similares a partir de una consulta, aplicando técnicas de **vectorización** y **similitud coseno**.

---

## 🧠 Descripción General

El sistema realiza las siguientes tareas:

1. **Recolección** de artículos científicos mediante *web scraping*.  
2. **Normalización** del texto (tokenización, lematización y eliminación de palabras vacías).  
3. **Representación vectorial** del texto utilizando modelos de frecuencia, binario y TF-IDF.  
4. **Comparación de similitud** entre artículos mediante el cálculo de la similitud coseno.  

Con esto, se logra una recuperación eficiente de los artículos más parecidos a una consulta dada (por título o resumen).

---

## ⚙️ Componentes Principales

### 1. Recolección de Artículos (`web_scrapping.py`)
Obtiene artículos recientes desde los siguientes repositorios:

- **arXiv:** Computation and Language, Computer Vision and Pattern Recognition, Cryptography and Security.  
- **PubMed:** Artículos de la sección *Trending*.

Datos recolectados:
- DOI  
- Título  
- Autores  
- Resumen  
- Sección o Revista  
- Fecha de publicación  

Los artículos se guardan en archivos CSV separados por tabulaciones.

---

### 2. Normalización del Texto

#### 🔹 Usando NLTK 
- Tokenización  
- Etiquetado gramatical (POS tagging)  
- Eliminación de *stop words* (artículos, preposiciones, conjunciones y pronombres)  
- Lematización con **WordNetLemmatizer**


---

### 3. Representación Vectorial (`vectorizacion.py`)
Genera representaciones numéricas de los textos mediante tres métodos:

- **Frecuencia (CountVectorizer)**  
- **Binaria (CountVectorizer con binary=True)**  
- **TF-IDF (TfidfVectorizer)**  

Rangos de n-gramas:
- Unigramas `(1,1)`  
- Bigramas `(2,2)`  
- Combinados `(1,3)`

Las representaciones se guardan como archivos `.pkl` para su reutilización.

---

### 4. Similitud de Documentos (`document_similarity 1.py`)
Calcula la **similitud coseno** entre los vectores de los documentos:

\[
\text{Similitud}(x, y) = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}}
\]

El sistema muestra los **10 documentos más similares** en orden descendente, permitiendo comparar artículos por su título o resumen.

---

## 🧩 Flujo de Trabajo

1. **Recolectar** artículos → `web_scrapping.py`  
2. **Normalizar** el corpus → `normalization_nltk_english.py` o `normalization_spacy_english.py`  
3. **Vectorizar** el texto → `vectorizacion.py`  
4. **Comparar** documentos → `document_similarity 1.py`  

---

## 🖥️ Interfaz

Se puede añadir una interfaz gráfica para facilitar:
- La recolección de artículos.  
- La normalización y vectorización del corpus.  
- La comparación y visualización de artículos similares.  

---

## 📈 Resultados Esperados

Archivos generados:
- `arxiv_raw_corpus.csv`, `pubmed_raw_corpus.csv`  
- `arxiv_normalized_corpus.csv`, `pubmed_normalized_corpus.csv`  
- Archivos `.pkl` de representaciones vectoriales  
- Archivos `.tsv` o `.csv` con los resultados de similitud  

---

## 🧰 Tecnologías Utilizadas

- **Python 3**
- **BeautifulSoup4**
- **Requests**
- **NLTK**
- **spaCy**
- **scikit-learn**
- **pandas**
- **pickle**

---

## 👩‍💻 Autores

Proyecto desarrollado con fines académicos e investigativos, orientado al **procesamiento de lenguaje natural** y **recuperación de información** a partir de repositorios científicos de libre acceso.

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia **MIT**, y su uso está permitido con fines académicos y de investigación.

