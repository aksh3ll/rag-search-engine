from collections import defaultdict
import os
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from lib.chunk_utils import chunk_sentences

SCORE_PRECISION = 4
CACHE_DIR = './cache/'
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'movie_embeddings.npy')
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'chunk_embeddings.npy')
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, 'chunk_metadata.json')

class SemanticSearch:

    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.documents: list = []
        self.document_map: dict = {}
        self.loaded = False

    def __repr__(self):
        return f"Model loaded: {self.model}\nMax sequence length: {self.model.max_seq_length}"
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError('The text is empty.')
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list):
        print("Building embeddings...")
        self.documents = documents
        data = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            data.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(data, show_progress_bar=True)
        self.save_embeddings()
        return self.embeddings

    def save_embeddings(self):
        np.save(EMBEDDINGS_PATH, self.embeddings)

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(EMBEDDINGS_PATH):
            print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) != len(self.documents):
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)
        self.loaded = True
        return self.embeddings
      
    
    def search(self, query: str, limit: int) -> list[dict]:
        if not self.loaded:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embed_query = self.generate_embedding(query)
        print(f"Generated embedding for query: {embed_query[:3]}... (shape: {embed_query.shape})")

        results = sorted([
            (cosine_similarity(embed_query, embed_doc), self.documents[pos])
            for pos, embed_doc in enumerate(self.embeddings)
        ], key=lambda item: item[0], reverse=True)[:limit]
        print(f"Search returned {len(results)} results.")

        return [
            {'score': score, 'title': doc['title'], 'description': doc['description'], 'doc_id': doc['id']}
            for score, doc in results
        ]
    
    def cross_encode(self, pairs: list[list]) -> list:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        # `predict` returns a list of numbers, one for each pair
        return cross_encoder.predict(pairs)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict] = []
        self.total_chunks: int = 0

    def build_chunk_embeddings(self, documents: list):
        self.documents = documents
        all_chunks = []
        self.chunks_metadata = []
        for movie_idx, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
            if not doc['description']:
                continue

            doc_desc_chunks = chunk_sentences(doc['description'], 4, 1)
            for chunk in doc_desc_chunks:
                all_chunks.append(' '.join(chunk))

            for chunk_idx in range(len(doc_desc_chunks)):
                self.chunks_metadata.append(
                    {'movie_idx': movie_idx, 'chunk_idx': chunk_idx, 'total_chunks': len(all_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({"chunks": self.chunks_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        self.loaded = True
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for movie_idx, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            print(f"Loading chunk embeddings from {CHUNK_EMBEDDINGS_PATH} and metadata from {CHUNK_METADATA_PATH}...")
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                self.chunks_metadata = metadata['chunks']
                self.total_chunks = metadata['total_chunks']
            self.loaded = True
        else:
            print("Chunk embeddings or metadata not found. Building chunk embeddings...")
            self.build_chunk_embeddings(documents)    
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if not self.loaded:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embed_query = self.generate_embedding(query)

        chunk_score = []
        for pos, chunk_embed in enumerate(self.chunk_embeddings):
            chunk_metadata = self.chunks_metadata[pos]
            chunk_score.append((chunk_metadata['chunk_idx'], chunk_metadata['movie_idx'], cosine_similarity(embed_query, chunk_embed)))

        movie_score = defaultdict(float)
        for chunk_idx, movie_idx, score in chunk_score:
            if score > movie_score[movie_idx]:
                movie_score[movie_idx] = score

        results = sorted(movie_score.items(), key=lambda item: item[1], reverse=True)[:limit]
        metadata = {}
        return [{
            "id": self.documents[doc_idx]['id'],
            "title": self.documents[doc_idx]['title'],
            "document": self.documents[doc_idx]['description'][:100],
            "score": round(score, SCORE_PRECISION),
            "metadata": metadata or {}
        } for doc_idx, score in results]


def verify_model():
    print(SemanticSearch())


def embed_text(text: str):
    embedding = SemanticSearch().generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def load_embeddings():
    ss = SemanticSearch()
    with open('data/movies.json') as f:
        documents = json.load(f)['movies']
    ss.load_or_create_embeddings(documents)
    return ss


def verify_embeddings():
    ss = load_embeddings()
    print(f"Number of docs:   {len(ss.documents)}")
    print(f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query: str, limit: int=5):
    results = load_embeddings().search(query, limit)
    for pos, result in enumerate(results, start=1):
        print(f'{pos}. {result['title']} (score: {result['score']})\n\t{result['description'][:40]}...')
