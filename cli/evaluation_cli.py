import argparse
import json
from lib.hybrid_search import HybridSearch, normalize_scores
from data_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open('data/golden_dataset.json', 'r') as f:
        golden_dataset = json.load(f)

    hs = HybridSearch(load_movies())
    k = 60
    print(f'k = {limit}')
    print()

    for test in golden_dataset['test_cases']:
        query: str = test['query']
        relevant_docs: list[str] = test['relevant_docs']
        results = hs.rrf_search(query, k, limit)
        titles = [doc['title'] for doc in results]
        total_retrieved: int = len(titles)
        relevant_retrieved: int = len([t for t in titles if t in relevant_docs])
        total_relevant = len(relevant_docs)
        precision: float = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f'- Query: {query}')
        print(f'  - Precision@{limit}: {precision:.4f}')
        print(f'  - Recall@{limit}: {recall:.4f}')
        print(f'  - F1 Score: {f1:.4f}')
        print(f'  - Retrieved: {', '.join(titles)}')
        print(f'  - Relevant: {', '.join(relevant_docs)}')
        print()
     

if __name__ == "__main__":
    main()
