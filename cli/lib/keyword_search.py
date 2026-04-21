from collections import defaultdict, Counter
import math
import os
import pickle
from typing import Self

from tokenizer import to_tokens


CACHE_DIR = './cache/'
INDEX_PATH = os.path.join(CACHE_DIR, 'index.pkl')
DOCMAP_PATH = os.path.join(CACHE_DIR, 'docmap.pkl')
TF_PATH = os.path.join(CACHE_DIR, 'term_frequencies.pkl')
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.doc_lengths: dict[int, int] = {}
        self.term_frequencies = Counter()

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = to_tokens(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[(doc_id, token)] += 1
        self.doc_lengths[doc_id] = len(tokens)
    
    def __get_avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list:
        result = self.index[term.lower()]
        return list(sorted(result)) if result else []
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = to_tokens(term)
        if len(tokens) > 1:
            raise Exception('get_tf only accept one token as term')
        return self.term_frequencies[(doc_id, tokens[0])]
    
    def get_idf(self, term: str) -> float:
        tokens = to_tokens(term)
        if len(tokens) > 1:
            raise Exception('get_idf only accept one token as term')
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(tokens[0]))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1)) 

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = to_tokens(term)
        if len(tokens) > 1:
            raise Exception('get_bm25_idf only accept one token as term')

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(tokens[0]))

        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term, k1: float=BM25_K1, b: float=BM25_B) -> float:
        tokens = to_tokens(term)
        if len(tokens) > 1:
            raise Exception('get_bm25_tf only accept one token as term')
        tf = self.get_tf(doc_id, term)

        # Length normalization factor
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

    def bm25_search(self, query: str, limit: int) -> list:
        tokens = to_tokens(query)

        scores = sorted([
            (doc_id, sum([self.get_bm25(doc_id, token) for token in tokens]))
            for doc_id in self.doc_lengths
        ], key=lambda item: item[1], reverse=True)[:limit]
        return [(doc_id, self.docmap[doc_id]['title'], score) for doc_id, score in scores]

    def build(self, documents) -> Self:
        for m in documents:
            self.docmap[m['id']] = m
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
        return self

    def save(self) -> Self:
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
        return self

    def load(self) -> Self:
        if not os.path.isdir(CACHE_DIR):
            raise Exception('Cache folder does not exists, you should build your index first')
        for path in [INDEX_PATH, DOCMAP_PATH, TF_PATH, DOC_LENGTHS_PATH]:
            if not os.path.exists(path):
                raise Exception(f'{path} file does not exists, you should build your index first')
        
        with open(INDEX_PATH, 'rb') as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(TF_PATH, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, 'rb') as f:
            self.doc_lengths = pickle.load(f)
        return self
