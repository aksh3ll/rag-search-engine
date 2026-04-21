import argparse
import math
import sys

from data_utils import load_movies
from tokenizer import to_tokens
from cli.lib.keyword_search import InvertedIndex, BM25_K1, BM25_B


def matching(query: str, index: InvertedIndex, max_values: int = 5) -> list:
    result = []
    for token in to_tokens(query):
        for doc_id in index.get_documents(token):
            result.append((doc_id, index.docmap[doc_id]['title']))
            if len(result) >= max_values:
                break
    return result

def load_index() -> InvertedIndex:
    try:
        return InvertedIndex().load()
    except Exception as err:
        print(f'Error: {err}')
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build and save the movies index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a specific term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a specific term in all documents")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")
    tfidf_parser = subparsers.add_parser("tfidf", help="Get term frequency and inverse document frequency for a specific term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get frequency for")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case 'build':
            InvertedIndex().build(load_movies()).save()
        case 'search':
            print(f'Searching for: {args.query}')
            for id, title in matching(args.query, load_index()):
                print(f'{id}. {title}')
        case 'tf':
            print(f'Searching term frequency for: {args.term} in document {args.doc_id}: {load_index().get_tf(args.doc_id, args.term)}')
        case 'idf':
            print(f"Inverse document frequency of '{args.term}': {load_index().get_idf(args.term):.2f}")
        case 'tfidf':
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {load_index().get_tfidf(args.doc_id, args.term):.2f}")
        case 'bm25idf':
            print(f"BM25 IDF score of '{args.term}': {load_index().get_bm25_idf(args.term):.2f}")
        case 'bm25tf':
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {load_index().get_bm25_tf(args.doc_id, args.term, args.k1, args.b):.2f}")
        case 'bm25search':
            print(f'Searching with bm25search for: {args.query}')
            for id, title, score in load_index().bm25_search(args.query, 5):
                print(f'({id}) {title} - Score: {score:.2f}')
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
