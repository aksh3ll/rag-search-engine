#!/usr/bin/env python3
import argparse
import re
from lib.semantic_search import (SemanticSearch, ChunkedSemanticSearch,
                                 verify_model, embed_text, verify_embeddings,
                                 embed_query_text, search)
from data_utils import load_movies
from lib.chunk_utils import chunk_words, chunk_sentences

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("verify", help="Verify current model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed text in current model")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    verif_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query in current model")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    search_parser = subparsers.add_parser("search", help="Search a movie")
    search_parser.add_argument("query", type=str, help="Query for movie")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable limit parameter")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a string")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Tunable chunk size parameter")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Tunable overlap parameter")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Sementic chunk of a string")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Tunable max chunk size parameter")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Tunable overlap parameter")
    embed_chunks_parser = subparsers.add_parser('embed_chunks', help='Build the embedding chunks')
    search_chunked_parser = subparsers.add_parser('search_chunked', help='Search a movie by chunks')
    search_chunked_parser.add_argument("query", type=str, help="Query for movie")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable limit parameter")

    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()
        case 'embed_text':
            embed_text(args.text)
        case 'verify_embeddings':
            verify_embeddings()
        case 'embedquery':
            embed_query_text(args.query)
        case 'search':
            search(args.query, args.limit)
        case 'chunk':
            chunks = chunk_words(args.text, args.chunk_size, args.overlap)
            print(f'Chunking {len(args.text)} characters')
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {' '.join(chunk)}")
        case 'semantic_chunk':
            chunks = chunk_sentences(args.text, args.max_chunk_size, args.overlap)
            print(f'Semantically chunking {len(args.text)} characters')
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {' '.join(chunk)}")
        case 'embed_chunks': 
            embeddings = ChunkedSemanticSearch().load_or_create_chunk_embeddings(load_movies())
            print(f"Generated {len(embeddings)} chunked embeddings")
        case 'search_chunked':
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(load_movies())
            results = css.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
