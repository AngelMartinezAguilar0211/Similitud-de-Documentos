import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import pandas as pd
import os
import sys
import re
import io

class NoOpSpinner:
    def start(self, *_, **__): 
        pass
    def stop(self): 
        pass

def merge_section_journal(df):
    sec_col = None
    jour_col = None
    for c in df.columns:
        if c.lower() == "section":
            sec_col = c
        if c.lower() == "journal":
            jour_col = c

    sec_series = df[sec_col].astype(str) if sec_col in df.columns else pd.Series([""] * len(df))
    jour_series = df[jour_col].astype(str) if jour_col in df.columns else pd.Series([""] * len(df))

    sec_series = sec_series.replace({"nan": ""})
    jour_series = jour_series.replace({"nan": ""})


    combined = []
    for s, j in zip(sec_series, jour_series):
        s = s.strip()
        j = j.strip()
        if s and j:
            combined.append(f"{s} • {j}")
        elif s:
            combined.append(s)
        elif j:
            combined.append(j)
        else:
            combined.append("")
    df["Section"] = combined
    return df

def read_any_csv(path, source_name):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path, sep=",")
    df["Source"] = source_name
    return df

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def run_command(command):
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        full_cmd = [sys.executable] + command
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return f"Error executing command:\n{e}"


def parse_similarities_text(text):
    try:
        df_try = pd.read_csv(io.StringIO(text), sep="\t", header=None)
        if df_try.shape[1] >= 2:
            if df_try.shape[1] >= 5:
                df_try = df_try.iloc[:, :5]
                df_try.columns = ["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"]
                with pd.option_context('mode.chained_assignment', None):
                    df_try["Similarity"] = pd.to_numeric(df_try["Similarity"], errors="coerce")
                df_try = df_try.dropna(subset=["Similarity"]).sort_values("Similarity", ascending=False)
                return df_try.head(10)
            else:
                df_try = df_try.iloc[:, :2]
                df_try.columns = ["Corpus doc", "Similarity"]
                with pd.option_context('mode.chained_assignment', None):
                    df_try["Similarity"] = pd.to_numeric(df_try["Similarity"], errors="coerce")
                df_try = df_try.dropna(subset=["Similarity"]).sort_values("Similarity", ascending=False)
                df_try["Vector type"] = ""
                df_try["Features"] = ""
                df_try["Comparison"] = ""
                df_try = df_try[["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"]]
                return df_try.head(10)
    except Exception:
        pass
    rows = []
    sim_pat = re.compile(r"(?:sim(?:ilarity)?[:=]?\s*)?([0-1](?:[.,]\d+)?)", re.IGNORECASE)
    doc_pat = re.compile(r"(?:doc(?:ument)?|file|corpus)[:=]?\s*([^\t\n]+)", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue

        sim_match = sim_pat.search(line)
        sim_val = None
        if sim_match:
            s = sim_match.group(1).replace(",", ".")
            try:
                sim_val = float(s)
            except Exception:
                sim_val = None

        doc_match = doc_pat.search(line)
        doc_val = None
        if doc_match:
            doc_val = doc_match.group(1).strip()
        else:
            if "sim" in line.lower():
                candidate = re.split(r"\s[\-\|]\s|\t", line)[0].strip()
                if not candidate.lower().startswith("sim"):
                    doc_val = candidate

        if sim_val is not None:
            vec = ""
            feat = ""
            comp = ""
            l = line.lower()
            if "tfidf" in l:
                vec = "tfidf"
            elif "frequency" in l:
                vec = "frequency"
            elif "binary" in l:
                vec = "binary"
            if "unigram" in l or "n1-1" in l:
                feat = "unigram"
            elif "bigram" in l or "n2-2" in l:
                feat = "bigram"
            if "title" in l and "abstract" in l:
                comp = "Title vs Abstract"
            elif "title" in l:
                comp = "Title"
            elif "abstract" in l:
                comp = "Abstract"

            rows.append([doc_val or "", vec, feat, comp, sim_val])

    if rows:
        df = pd.DataFrame(rows, columns=["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"])
        df = df.dropna(subset=["Similarity"]).sort_values("Similarity", ascending=False)
        return df.head(10)

    return pd.DataFrame(columns=["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"])


class PaginatedTable(ttk.Frame):
    def __init__(self, parent, columns, page_size=10):
        super().__init__(parent)
        self.columns = columns
        self.page_size = page_size
        self.data = None
        self.current_page = 0

        self.tree = ttk.Treeview(self, columns=self.columns, show="headings", height=self.page_size)
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=160, anchor="center")
        self.tree.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(self, style="TFrame")
        btn_frame.pack(pady=4)
        for symbol, cmd in zip(["⏮", "⏪", "⏩", "⏭"], [self.first_page, self.prev_page, self.next_page, self.last_page]):
            ttk.Button(btn_frame, text=symbol, command=cmd, width=4).pack(side="left", padx=3)

        self.show_message("Datos sin cargar")

    def show_message(self, msg):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree.insert("", "end", values=[msg] + [""] * (len(self.columns) - 1))

    def load_data(self, df):
        self.data = df
        self.current_page = 0
        self.refresh()

    def refresh(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        if self.data is None or self.data.empty:
            self.show_message("No data loaded yet — please run this process first.")
            return
        start = self.current_page * self.page_size
        end = start + self.page_size
        for _, row in self.data.iloc[start:end].iterrows():
            self.tree.insert("", "end", values=list(row))

    def next_page(self):
        if self.data is not None and (self.current_page + 1) * self.page_size < len(self.data):
            self.current_page += 1
            self.refresh()

    def prev_page(self):
        if self.data is not None and self.current_page > 0:
            self.current_page -= 1
            self.refresh()

    def first_page(self):
        self.current_page = 0
        self.refresh()

    def last_page(self):
        if self.data is not None:
            self.current_page = max(0, (len(self.data) - 1) // self.page_size)
            self.refresh()


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Practica II – Similitud de Documentos")
        self.state("zoomed")
        self.configure(bg="#1E1E1E")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1E1E1E", foreground="#D4D4D4", fieldbackground="#252526", font=("Segoe UI", 10))
        style.configure("TFrame", background="#252526")
        style.configure("TLabelframe", background="#252526", relief="ridge", borderwidth=1)
        style.configure("TLabelframe.Label", background="#252526", foreground="#D4D4D4")
        style.configure("TButton", background="#0A84FF", foreground="#FFFFFF", padding=6, relief="flat")
        style.map("TButton", background=[("active", "#1A94FF")])
        style.configure("TEntry", fieldbackground="#252526", foreground="#D4D4D4")
        style.configure("TCombobox", fieldbackground="#252526", background="#252526",
                        selectbackground="#252526", foreground="#D4D4D4")
        style.configure("Treeview", background="#252526", fieldbackground="#252526",
                        foreground="#D4D4D4", bordercolor="#3C3C3C")
        style.configure("Treeview.Heading", background="#2D2D2D", foreground="#E0E0E0")
        style.configure("TNotebook", background="#1E1E1E", borderwidth=0)
        style.configure("TNotebook.Tab", background="#D4D4D4", foreground="black")
        style.map("TNotebook.Tab", background=[("selected", "#0A84FF")], foreground=[("selected", "black")])

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.spinner = NoOpSpinner()

        self.tabs = {}
        for name in ["Recoleccion", "Normalizacion", "Vectorizacion", "Recuperacion"]:
            frame = ttk.Frame(notebook, style="TFrame")
            notebook.add(frame, text=name)
            self.tabs[name] = frame

        self.setup_collection()
        self.setup_normalization()
        self.setup_vectorization()
        self.setup_retrieval()

    def browse_file(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def run_with_spinner(self, command, console, on_finish=None):
        self.spinner.start(10)
        self.update()
        output = run_command(command)
        self.spinner.stop()
        console.delete("1.0", tk.END)
        console.insert(tk.END, output)
        if on_finish:
            on_finish()
        messagebox.showinfo("Process completed", "Process completed successfully.")

    # ---------- Tab 1: Recoleccion ----------
    def setup_collection(self):
        tab = self.tabs["Recoleccion"]
        tab.grid_rowconfigure(0, weight=0, minsize=160)   # Controles
        tab.grid_rowconfigure(1, weight=2, minsize=260)   # Tabla
        tab.grid_rowconfigure(2, weight=1, minsize=140)   # Consola
        tab.grid_columnconfigure(0, weight=1)

        # Controles
        controls = ttk.Labelframe(tab, style="TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 8))
        controls.grid_columnconfigure(1, weight=1)

        ttk.Label(controls, text="Repositorio:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.repo = ttk.Combobox(controls, values=["arxiv", "pubmed", "both"], state="readonly", width=15)
        self.repo.set("both")
        self.repo.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Salida ArXiv:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.arxiv_out = ttk.Entry(controls)
        self.arxiv_out.insert(0, "data/corpus/arxiv_raw_corpus.csv")
        self.arxiv_out.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        ttk.Label(controls, text="Salida PubMed:").grid(row=2, column=0, padx=6, pady=6, sticky="e")
        self.pubmed_out = ttk.Entry(controls)
        self.pubmed_out.insert(0, "data/corpus/pubmed_raw_corpus.csv")
        self.pubmed_out.grid(row=2, column=1, padx=6, pady=6, sticky="ew")

        run_f = ttk.Labelframe(controls, style="TLabelframe")
        run_f.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        ttk.Button(run_f, text="Ejecutar Recoleccion", command=self.run_collection).pack(side="left", padx=10, pady=6)
        

        # Tabla
        table_wrap = ttk.Labelframe(tab, style="TLabelframe")
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)
        self.table_col = PaginatedTable(table_wrap, ["DOI", "Titulo", "Autores", "Resumen", "Seccion", "Fecha", "Fuente"], page_size=10)
        self.table_col.grid(row=0, column=0, sticky="nsew")

        # Consola
        console_wrap = ttk.Labelframe(tab, style="TLabelframe")
        console_wrap.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 15))
        console_wrap.grid_rowconfigure(0, weight=1)
        console_wrap.grid_columnconfigure(0, weight=1)
        self.console_col = tk.Text(console_wrap, height=8, wrap="word", bg="#2D2D2D", fg="white")
        self.console_col.grid(row=0, column=0, sticky="nsew")

    def run_collection(self):
        ensure_parent_dir(self.arxiv_out.get())
        ensure_parent_dir(self.pubmed_out.get())
        repo = self.repo.get()
        cmd = ["-m", "scraper.cli", "--repo", repo,
               "--arxiv-out", self.arxiv_out.get(), "--pubmed-out", self.pubmed_out.get()]

        def on_finish():
            repo_sel = self.repo.get().strip().lower()
            dfs = []

            # Rutas de salida
            arxiv_path = self.arxiv_out.get().strip()
            pubmed_path = self.pubmed_out.get().strip()

            # Cargar según seleccion
            if repo_sel in ("arxiv", "both") and os.path.exists(arxiv_path):
                try:
                    dfs.append(read_any_csv(arxiv_path, "arxiv"))
                except Exception as e:
                    print(f"[Recoleccion] No se pudo leer arxiv: {e}")

            if repo_sel in ("pubmed", "both") and os.path.exists(pubmed_path):
                try:
                    dfs.append(read_any_csv(pubmed_path, "pubmed"))
                except Exception as e:
                    print(f"[Recoleccion] No se pudo leer pubmed: {e}")

            if not dfs:
                self.table_col.load_data(pd.DataFrame(columns=["DOI","Title","Authors","Abstract","Section","Date","Source"]))
                return

            df_all = pd.concat(dfs, ignore_index=True)

            df_all = merge_section_journal(df_all)

            for col in ["DOI","Title","Authors","Abstract","Section","Date","Source"]:
                if col not in df_all.columns:
                    df_all[col] = ""

            df_all = df_all[["DOI","Title","Authors","Abstract","Section","Date","Source"]]

            self.table_col.load_data(df_all)


        self.run_with_spinner(cmd, self.console_col, on_finish)

    # ---------- Tab 2: Normalizacion ----------
    def setup_normalization(self):
        tab = self.tabs["Normalizacion"]
        tab.grid_rowconfigure(0, weight=0, minsize=160)
        tab.grid_rowconfigure(1, weight=2, minsize=260)
        tab.grid_rowconfigure(2, weight=1, minsize=140)
        tab.grid_columnconfigure(0, weight=1)

        controls = ttk.Labelframe(tab, style="TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 8))
        controls.grid_columnconfigure(1, weight=1)

        ttk.Label(controls, text="Archivo de Entrada:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.norm_in = ttk.Entry(controls)
        self.norm_in.insert(0, "data/corpus/arxiv_raw_corpus.csv")
        self.norm_in.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(controls, text="Examinar", command=lambda: self.browse_file(self.norm_in)).grid(row=0, column=2, padx=6)

        ttk.Label(controls, text="Archivo de Salida:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.norm_out = ttk.Entry(controls)
        self.norm_out.insert(0, "data/corpus/arxiv_normalized_corpus.csv")
        self.norm_out.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        run_f = ttk.Labelframe(controls, style="TLabelframe")
        run_f.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        ttk.Button(run_f, text="Ejecutar Normalizacion", command=self.run_normalization).pack(side="left", padx=10, pady=6)
        

        table_wrap = ttk.Labelframe(tab, style="TLabelframe")
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)
        self.table_norm = PaginatedTable(table_wrap, ["DOI", "Título", "Resumen"], page_size=10)
        self.table_norm.grid(row=0, column=0, sticky="nsew")

        console_wrap = ttk.Labelframe(tab, style="TLabelframe")
        console_wrap.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 15))
        console_wrap.grid_rowconfigure(0, weight=1)
        console_wrap.grid_columnconfigure(0, weight=1)
        self.console_norm = tk.Text(console_wrap, height=8, wrap="word", bg="#2D2D2D", fg="white")
        self.console_norm.grid(row=0, column=0, sticky="nsew")

    def run_normalization(self):
        ensure_parent_dir(self.norm_out.get())
        cmd = ["-m", "normalization.normalization",
               "--input", self.norm_in.get(), "--output", self.norm_out.get()]

        def on_finish():
            if os.path.exists(self.norm_out.get()):
                df = pd.read_csv(self.norm_out.get(), sep="\t")
                self.table_norm.load_data(df.head(300))

        self.run_with_spinner(cmd, self.console_norm, on_finish)

    # ---------- Tab 3: Vectorizacion ----------
    def setup_vectorization(self):
        tab = self.tabs["Vectorizacion"]
        tab.grid_rowconfigure(0, weight=0, minsize=160)
        tab.grid_rowconfigure(1, weight=0, minsize=0)  
        tab.grid_rowconfigure(2, weight=1, minsize=180) 
        tab.grid_columnconfigure(0, weight=1)

        controls = ttk.Labelframe(tab, style="TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 8))
        controls.grid_columnconfigure(1, weight=1)

        ttk.Label(controls, text="Ruta Base:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.vec_base = ttk.Entry(controls)
        self.vec_base.insert(0, ".\\data")
        self.vec_base.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        ttk.Label(controls, text="Corpus:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.vec_corpus = ttk.Combobox(controls, values=["arxiv", "pubmed", "both"], state="readonly")
        self.vec_corpus.set("both")
        self.vec_corpus.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Field:").grid(row=2, column=0, padx=6, pady=6, sticky="e")
        self.vec_field = ttk.Combobox(controls, values=["Title", "Abstract", "Both"], state="readonly")
        self.vec_field.set("Both")
        self.vec_field.grid(row=2, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Representacion:").grid(row=3, column=0, padx=6, pady=6, sticky="e")
        self.vec_rep = ttk.Combobox(controls, values=["tfidf", "frequency", "binary", "all"], state="readonly")
        self.vec_rep.set("all")
        self.vec_rep.grid(row=3, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="N-grams:").grid(row=4, column=0, padx=6, pady=6, sticky="e")
        self.vec_ngrams = ttk.Combobox(controls, values=["unigram", "bigram", "both"], state="readonly")
        self.vec_ngrams.set("both")
        self.vec_ngrams.grid(row=4, column=1, padx=6, pady=6, sticky="w")

        run_f = ttk.Labelframe(controls, style="TLabelframe")
        run_f.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        ttk.Button(run_f, text="Ejecutar Vectorizacion", command=self.run_vectorization).pack(side="left", padx=10, pady=6)
        

        console_wrap = ttk.Labelframe(tab, style="TLabelframe")
        console_wrap.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 15))
        console_wrap.grid_rowconfigure(0, weight=1)
        console_wrap.grid_columnconfigure(0, weight=1)
        self.console_vec = tk.Text(console_wrap, height=10, wrap="word", bg="#2D2D2D", fg="white")
        self.console_vec.grid(row=0, column=0, sticky="nsew")

    def run_vectorization(self):
        cmd = [
            "-m", "representation.vectorize",
            "--basepath", self.vec_base.get(),
            "--corpus", self.vec_corpus.get(),
            "--field", self.vec_field.get(),
            "--rep", self.vec_rep.get(),
            "--ngrams", self.vec_ngrams.get()
        ]
        self.run_with_spinner(cmd, self.console_vec)

    # ---------- Tab 4: Recuperacion ----------
    def setup_retrieval(self):
        tab = self.tabs["Recuperacion"]
        tab.grid_rowconfigure(0, weight=0, minsize=170)
        tab.grid_rowconfigure(1, weight=2, minsize=220)
        tab.grid_rowconfigure(2, weight=1, minsize=140)
        tab.grid_columnconfigure(0, weight=1)

        controls = ttk.Labelframe(tab, style="TLabelframe")
        controls.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 8))
        controls.grid_columnconfigure(1, weight=1)

        ttk.Label(controls, text="Archivo de Consulta:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.sim_file = ttk.Entry(controls)
        self.sim_file.insert(0, "data/")
        self.sim_file.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(controls, text="Examinar", command=lambda: self.browse_file(self.sim_file)).grid(row=0, column=2, padx=6)

        ttk.Label(controls, text="Campo:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.sim_field = ttk.Combobox(controls, values=["Title", "Abstract"], state="readonly")
        self.sim_field.set("Abstract")
        self.sim_field.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Vector:").grid(row=2, column=0, padx=6, pady=6, sticky="e")
        self.sim_vector = ttk.Combobox(controls, values=["tfidf", "frequency", "binary"], state="readonly")
        self.sim_vector.set("tfidf")
        self.sim_vector.grid(row=2, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="N-gramas:").grid(row=3, column=0, padx=6, pady=6, sticky="e")
        self.sim_ngrams = ttk.Combobox(controls, values=["unigram", "bigram", "both"], state="readonly")
        self.sim_ngrams.set("unigram")
        self.sim_ngrams.grid(row=3, column=1, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Ruta base:").grid(row=4, column=0, padx=6, pady=6, sticky="e")
        self.sim_base = ttk.Entry(controls)
        self.sim_base.insert(0, ".")
        self.sim_base.grid(row=4, column=1, padx=6, pady=6, sticky="ew")

        ttk.Label(controls, text="Archivo de salida:").grid(row=5, column=0, padx=6, pady=6, sticky="e")
        self.sim_out = ttk.Entry(controls)
        self.sim_out.insert(0, "data/similar_articles")
        self.sim_out.grid(row=5, column=1, padx=6, pady=6, sticky="ew")

        run_f = ttk.Labelframe(controls, style="TLabelframe")
        run_f.grid(row=6, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        ttk.Button(run_f, text="Ejecutar Devolucion", command=self.run_retrieval).pack(side="left", padx=10, pady=6)
        

        table_wrap = ttk.Labelframe(tab, style="TLabelframe")
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)
        self.table_sim = PaginatedTable(table_wrap, ["Documento", "Vector", "Caracteristicas", "Comparacion", "Similitud"], page_size=10)
        self.table_sim.grid(row=0, column=0, sticky="nsew")

        console_wrap = ttk.Labelframe(tab, style="TLabelframe")
        console_wrap.grid(row=2, column=0, sticky="nsew", padx=20, pady=(8, 15))
        console_wrap.grid_rowconfigure(0, weight=1)
        console_wrap.grid_columnconfigure(0, weight=1)
        self.console_sim = tk.Text(console_wrap, height=8, wrap="word", bg="#2D2D2D", fg="white")
        self.console_sim.grid(row=0, column=0, sticky="nsew")

    def run_retrieval(self):
        ensure_parent_dir(self.sim_out.get())
        cmd = [
            "-m", "similarities.retrieve_similar_articles",
            "--file", self.sim_file.get(),
            "--field", self.sim_field.get(),
            "--vector", self.sim_vector.get(),
            "--ngrams", self.sim_ngrams.get(),
            "--basepath", self.sim_base.get(),
            "--output", self.sim_out.get()
        ]

        def on_finish():
            out_txt = self.sim_out.get().strip()
            base, ext = os.path.splitext(out_txt)
            out_tsv = base + ".tsv"

            if os.path.exists(out_tsv):
                try:
                    try:
                        df = pd.read_csv(out_tsv, sep="\t")
                    except Exception:
                        df = pd.read_csv(out_tsv, sep="\t", header=None)
                        if df.shape[1] == 5:
                            df.columns = ["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"]
                    if "Similarity" in df.columns:
                        df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
                        df = df.dropna(subset=["Similarity"]).sort_values("Similarity", ascending=False)
                    self.table_sim.load_data(df.head(10))
                    return
                except Exception:
                    pass

            if os.path.exists(out_txt):
                try:
                    try:
                        df = pd.read_csv(out_txt, sep="\t")
                    except Exception:
                        df = pd.read_csv(out_txt, sep="\t", header=None)
                        if df.shape[1] == 5:
                            df.columns = ["Corpus doc", "Vector type", "Features", "Comparison", "Similarity"]
                    if "Similarity" in df.columns:
                        df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
                        df = df.dropna(subset=["Similarity"]).sort_values("Similarity", ascending=False)
                    self.table_sim.load_data(df.head(10))
                    return
                except Exception:
                    pass

            text_payload = ""
            if os.path.exists(out_txt):
                try:
                    with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
                        text_payload = f.read()
                except Exception:
                    try:
                        with open(out_txt, "r", encoding="cp1252", errors="replace") as f:
                            text_payload = f.read()
                    except Exception:
                        text_payload = ""

            df = parse_similarities_text(text_payload) if text_payload else pd.DataFrame()
            if not df.empty:
                self.table_sim.load_data(df)
            else:
                self.table_sim.load_data(pd.DataFrame([{
                    "Corpus doc": "No structured results",
                    "Vector type": "",
                    "Features": "",
                    "Comparison": "",
                    "Similarity": ""
                }]))


        self.run_with_spinner(cmd, self.console_sim, on_finish)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
