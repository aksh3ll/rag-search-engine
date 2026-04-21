import os
from collections import defaultdict
from .keyword_search import InvertedIndex, INDEX_PATH
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500)
        bm25_scores_normalized = normalize_scores([r[2] for r in bm25_results])
        bm25_results_normalized = [(bm25_results[i][0], bm25_scores_normalized[i]) for i in range(len(bm25_results))]

        semantic_results = self.semantic_search.search_chunks(query, 500)
        semantic_scores_normalized = normalize_scores([r['score'] for r in semantic_results])
        semantic_results_normalized = [(semantic_results[i]['id'], semantic_scores_normalized[i]) for i in range(len(semantic_results))]

        merge: dict[int, dict] = defaultdict(lambda: {"bm25_score": 0.0, "semantic_score": 0.0})
        for doc_id, score in bm25_results_normalized:
            merge[doc_id]['bm25_score'] = score
        for doc_id, score in semantic_results_normalized:
            merge[doc_id]['semantic_score'] = score
        for doc_id in merge:
            merge[doc_id]['hybrid_score'] = hybrid_score(merge[doc_id]['bm25_score'], merge[doc_id]['semantic_score'])

        results = sorted(merge.items(), key=lambda item: item[1]['hybrid_score'], reverse=True)[:limit]

        return [
            {**r, 'title': self.document_map[doc_id]['title'], 'description': self.document_map[doc_id]['description']}
            for doc_id, r in results
        ]

    def rrf_search(self, query: str, k: int, limit=10) -> list[dict]:
        bm25_results = self._bm25_search(query, 500)
        bm25_results_normalized = [(res[0], i) for i, res in enumerate(bm25_results, start=1)]

        semantic_results = self.semantic_search.search_chunks(query, 500)
        semantic_results_normalized = [(res['id'], i) for i, res in enumerate(semantic_results, start=1)]

        merge: dict[int, dict] = defaultdict(lambda: {"bm25_rank": 9999, "semantic_rank": 9999})
        for doc_id, rank in bm25_results_normalized:
            merge[doc_id]['bm25_rank'] = rank
        for doc_id, rank in semantic_results_normalized:
            merge[doc_id]['semantic_rank'] = rank
        for doc_id in merge:
            merge[doc_id]['hybrid_score'] = hybrid_score(rrf_score(merge[doc_id]['bm25_rank'], k), rrf_score(merge[doc_id]['semantic_rank'], k))

        results = sorted(merge.items(), key=lambda item: item[1]['hybrid_score'], reverse=True)[:limit]

        return [
            {**r, 'doc_id': doc_id, 'title': self.document_map[doc_id]['title'], 'description': self.document_map[doc_id]['description']}
            for doc_id, r in results
        ]
    
    def cross_encode(self, pairs: list[list]) -> list:
        return self.semantic_search.cross_encode(pairs)


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]):
    if not scores:
        return []
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [1.0 for i in range(len(scores))]
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]
