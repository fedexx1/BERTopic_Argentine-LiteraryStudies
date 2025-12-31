import os
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk.corpus import stopwords
import unicodedata
import re
import json
import pandas as pd
import numpy as np



#PART 1: Documents pre-processing


nltk.download('stopwords', quiet=True)

# Custom stopwords to filter unwanted noise, specific to out corpus
spanish_stopwords = stopwords.words('spanish')
custom_stopwords = [
    # Metadata:
     "the", "issn", "http", "licencia", "anclajes", "anclaje", "ANCLAJES", "año", "and", "cuyo", 
     "permitir", "bajo", "creative", "commons", "caso", "siglo", "enero", "febrero", "marzo", 
     "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", 
     "diciembre", "xix", "xx", "xxi", "cuadernos", "auster", "eissn", "saga", "orbis", 
     "tertius", "cilha", "ano", "interfaz", "tipo", "ejemplo", "rubir", "rubir dario", 
     "casal", "aires", "martel", "maromero", "chileno", "bync", "cuaderno bync", "marin", 
     "castelnouovo", "mendoza", "eneroabril", "pdf generado", "septiembrediciembre", "pdf", 
     "generado", "cuaderno num", "num", "gagini", "hamed", "uruguay",
     "cuaderno international", "cuaderno", "international", "mayoagosto", "mendozar argentina", 
     "comparado volumen", "cultura comparado volumen", "comparado", "especialnoviembrir", 
     "volumen especialnoviembrir", "celaya", "sujo", "cuaderno junioseptiembre", "junioseptiembre", 
     "prejo", "benitez rojo", "benitez", "that", "rizal", "jodemir", "atargull", "mtp", 
    
    # General terms: 
     "editorial", "novela", "hombre", "literatura", "relato", "escritor", "lectura", "escritura", 
     "literario", "obra", "vida", "libro", "texto", "revista", "forma", "argentino", "historia", 
     "mundo", "espacio", "experiencia", "tiempo", "palabra", "lugar", "estudio", "relación", 
     "personaje", "mujer",
     "lector","autor", "ciencia", "dario", "juego" 
]

combined_stopwords = list(set(spanish_stopwords + custom_stopwords))

# Load Spanish language model
nlp = spacy.load("es_core_news_lg")


# Normalization and tokenization of documents

def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower()

def reconstruct_hyphenated_words(text):
    pattern = re.compile(r'(\w+)-\s*(\w+)')
    return pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)

def tokenize_text(text):
    text = reconstruct_hyphenated_words(text)
    text = normalize_text(text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_space and not token.is_stop and not token.is_punct 
              and token.lemma_.lower() not in combined_stopwords 
              and not token.like_num and len(token.text) > 2 and token.is_alpha]
    return tokens


# Directory processing, returning tokenized documents
def process_directory(directory):
    filenames = []
    documents = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    tokens = tokenize_text(text)
                    filenames.append(file_name)
                    documents.append(" ".join(tokens))
    return filenames, documents


# Load and process documents
directory_path = "user_path"
filenames, documents = process_directory(directory_path)


# Load articles metadata
from pathlib import Path as _Path
metadata_path = _Path("metadata_map.json")
if metadata_path.exists():
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
else:
    meta = {}

years = []
journals = []
for fn in filenames:
    info = meta.get(fn, {})
    y = info.get("publication_year")
    years.append(pd.to_datetime(f"{int(y)}-01-01") if y else pd.NaT)
    journals.append(info.get("journal", "Unknown"))



#PART 2: Topic modeling with BERTopic

# Generate embeddings using multilingual sentence transformer
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = embedding_model.encode(documents, batch_size=32, show_progress_bar=True)

# Dimensionality reduction: PCA (Principal Component Analysis) followed by UMAP (Uniform Manifold Approximation and Projection)
pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(embeddings)

umap_model = UMAP(n_neighbors=15, min_dist=0.0)
umap_embeddings = umap_model.fit_transform(pca_embeddings)

# Vectorization with CountVectorizer
count_vectorizer = CountVectorizer(
    stop_words=combined_stopwords,
    ngram_range=(1, 2),
    min_df=2
)

# Class-based TF-IDF with √TF-BM25(IDF) weighting (Borcin 2024)
ctfidf_model = ClassTfidfTransformer(
    bm25_weighting=True,
    reduce_frequent_words=True
)

# BERTopic model configuration
bertopic = BERTopic(
    embedding_model=None,
    umap_model=umap_model,
    hdbscan_model=hdbscan.HDBSCAN(
        min_cluster_size=100, 
        min_samples=10,
        prediction_data=True
    ),
    vectorizer_model=count_vectorizer,
    ctfidf_model=ctfidf_model,
    top_n_words=10
)

# Fit model
topics, probs = bertopic.fit_transform(documents, embeddings=umap_embeddings)


# Output directory for all results
OUTPUT_DIR = "main_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[created] {OUTPUT_DIR}/")

# Save outputs
topics_info = bertopic.get_topic_info()
topics_info.to_excel(os.path.join(OUTPUT_DIR, 'topics_info.xlsx'), index=False)

assignments_df = pd.DataFrame({
    "filename": filenames,
    "topic": topics,
    "year": years,
    "journal": journals
})

assignments_df.to_excel(os.path.join(OUTPUT_DIR, "topic_assignments.xlsx"), index=False)
bertopic.save(os.path.join(OUTPUT_DIR, "bertopic_model"))



#PART 3: visualizations



# Visualization 1: Hierarchical clustering

try:
    fig = bertopic.visualize_hierarchy()
    fig.write_html(os.path.join(OUTPUT_DIR, "hierarchical_clustering.html"))
    print(f"✓ Saved: hierarchical_clustering.html")
except Exception as e:
    print(f"✗ Could not generate hierarchical clustering: {e}")


# Visualization 2: Topics over time

try:
    # Filter out documents without years
    valid_indices = [i for i, y in enumerate(years) if pd.notna(y)]
    valid_docs = [documents[i] for i in valid_indices]
    valid_timestamps = [years[i] for i in valid_indices]
    valid_topics = [topics[i] for i in valid_indices]

    # Create topics over time
    topics_over_time = bertopic.topics_over_time(
        valid_docs,
        valid_timestamps,
        topics=valid_topics,
        nr_bins=20
    )

    fig = bertopic.visualize_topics_over_time(topics_over_time)
    fig.write_html(os.path.join(OUTPUT_DIR, "topics_over_time.html"))
    print(f"✓ Saved: topics_over_time.html")

    # Save the data
    topics_over_time.to_excel(os.path.join(OUTPUT_DIR, "topics_over_time_data.xlsx"), index=False)
    print(f"✓ Saved: topics_over_time_data.xlsx")
except Exception as e:
    print(f"✗ Could not generate topics over time: {e}")


# Visualization 3: Topics per class (journals)

try:
    topics_per_class = bertopic.topics_per_class(documents, classes=journals)
    fig = bertopic.visualize_topics_per_class(topics_per_class)
    fig.write_html(os.path.join(OUTPUT_DIR, "topics_per_journal.html"))
    print(f"✓ Saved: topics_per_journal.html")

    # Save the data
    topics_per_class.to_excel(os.path.join(OUTPUT_DIR, "topics_per_journal_data.xlsx"), index=False)
    print(f"✓ Saved: topics_per_journal_data.xlsx")
except Exception as e:
    print(f"✗ Could not generate topics per class: {e}")

