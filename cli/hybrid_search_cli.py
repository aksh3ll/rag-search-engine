import argparse
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.gemini_utils import enhance_query, rerank_individual, rerank_batch, gemini_evaluate
from data_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="normalize")
    normalize_parser.add_argument("weights", type=float, nargs="+", help="List of float, e.g. 0.2 0.8")
    weighted_search_parser = subparsers.add_parser('weighted-search', help='Weighted Search')
    weighted_search_parser.add_argument("query", type=str, help="Query to embed")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=0.5, help="Tunable alpha")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable limit parameter") 
    rrf_search_parser = subparsers.add_parser('rrf-search', help='RRF Search')
    rrf_search_parser.add_argument("query", type=str, help="Query to embed")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=60, help="Tunable k")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable limit parameter")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", 'expand'], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", 'batch', 'cross_encoder'], help='Rerank method')
    rrf_search_parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable evaluation")

    args = parser.parse_args()

    match args.command:
        case 'weighted-search':
            results = HybridSearch(load_movies()).weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"  {result['description'][:100]}...\n")
            pass
        case 'rrf-search':
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)
        case 'normalize':
            if isinstance(args.weights, list):
                for w in normalize_scores(args.weights):
                    print(f'# {w:.4f}')
        case _:
            parser.print_help()

def rrf_search(query: str, k: int = 60, limit: int = 5, enhance: str = '', rerank_method: str = '', evaluate: bool = False):
    print(f'query: {query}, k: {k}, limit: {limit}, enhance: {enhance}, rerank_method: {rerank_method}, evaluate: {evaluate}')

    if enhance in ('spell', 'rewrite', 'expand'):
        enhanced_query = enhance_query(enhance, query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query
    else:
        query = query

    print(f'method: {enhance}, enhanced query: {query}')

    if rerank_method in ('individual', 'batch', 'cross_encoder'):
        hs = HybridSearch(load_movies())
        results = hs.rrf_search(query, k, limit * 3)

        for i, r in enumerate(results, start=1):
            print(f"- {i}. {r['title']}, hybrid score: {r['hybrid_score']:.4f} keyword: {r['bm25_rank']:.4f} semantic: {r['semantic_rank']:.4f}")

        print(f"Re-ranking top {limit} results using {rerank_method} method...")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")

        match rerank_method:
            case 'individual':
                rerank_results: list[dict] = sorted([
                    {**result, 'rerank_score': rerank_individual(query, result)}
                    for result in results
                ], key=lambda item: item['rerank_score'], reverse=True)[:limit]
            case 'batch':
                rerank_scores = rerank_batch(query, results)
                results_dict = {result['doc_id']: result for result in results}
                rerank_results: list[dict] = sorted([
                    {**results_dict[doc_id], 'rerank_score': i}
                    for i, doc_id in enumerate(rerank_scores, start=1)
                ], key=lambda item: item['rerank_score'], reverse=False)[:limit]
            case 'cross_encoder':
                pairs = []
                for doc in results:
                    pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
                scores = hs.cross_encode(pairs)

                for i in range(len(results)):
                    print(f"- {i + 1 }. {results[i]['title']}, cross score: {scores[i]:.4f}")

                rerank_results: list[dict] = sorted([
                    {**result, 'cross_score': scores[i]}
                    for i, result in enumerate(results)
                ], key=lambda item: item['cross_score'], reverse=True)[:limit]
        

        for i, result in enumerate(rerank_results, start=1):
            print(f"{i}. {result['title']}")
            if 'rerank_score' in result:
                print(f"   Re-rank Score: {result['rerank_score']}/10")
            elif 'cross_score' in result:
                print(f"   Cross Encoder Score: {result['cross_score']:.3f}")
            print(f"   RRF Score: {result['hybrid_score']:.3f}")
            print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
            print(f"   {result['description'][:100]}...\n")
    else:
        results = HybridSearch(load_movies()).rrf_search(query, k, limit)
        printed_results: list[str] = []
        for i, result in enumerate(results, start=1):
            printed_results.append(f"""
{i}. {result['title']}
  Hybrid Score: {result['hybrid_score']:.3f}
  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}
  {result['description'][:100]}...
""")
    
        if evaluate:
            evaluations = gemini_evaluate(query, printed_results)
            evaluated_results : list[dict] = sorted([
                    {**result, 'evaluation': evaluations[i]}
                    for i, result in enumerate(results)
                ], key=lambda item: item['evaluation'], reverse=True)
            
            for i, result in enumerate(evaluated_results, start=1):
                print(f'{i}. {result['title']}: {result['evaluation']}/3')
        else:
            print(''.join(printed_results))


if __name__ == "__main__":
    main()
